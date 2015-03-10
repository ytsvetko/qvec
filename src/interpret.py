#!/usr/bin/env python3

"""
./interpret.py --in_vectors <filename> --interpretations <filename> --out_vectors  <filename>

"""
import argparse
import math
import numpy as np
import json
from gurobipy import *
from scipy.stats.stats import pearsonr 
from scipy import spatial
import timeit
import gzip
import sys
import random

parser = argparse.ArgumentParser()
parser.add_argument("--in_vectors", default="../data/en-svd-de-64.txt")
parser.add_argument("--interpretations", nargs='+', default="../data/supersenses/wn_noun.supersneses")
parser.add_argument("--out_file", default="model.sol")
parser.add_argument("--distance_metric", default="abs_correlation", help="correlation, abs_correlation, cosine, heuristic1")
parser.add_argument("--distance_metric_threshold", type=float, default=0.1, help="minimal distance")
parser.add_argument("--auto_threshold", action='store_true')
parser.add_argument("--optimization_direction", default="MAXIMIZE", help="MAXIMIZE, MINIMIZE")
parser.add_argument("--verbose", action='store_true')
args = parser.parse_args()


class Matrix(object):
  def __init__(self):
    self.matrix = {} #key - word; value = dict with key - column num, value - val
    self.vocab = set()
    self.number_of_columns = 0

  def Column(self, dim, vocab):
    #return dimension based on vocab
    column = []
    for word in sorted(vocab):
      column.append(self.matrix[word].get(dim, 0.0))
    return column

  def __repr__(self):
    result = []
    for word in sorted(self.vocab):
      line = [word]
      features = self.matrix[word]
      for col in range(self.number_of_columns):
        line.append(str(features.get(col, 0.0)))
      result.append(" ".join(line))
    return "\n".join(result)
    
class OracleMatrix(Matrix):
  def __init__(self):
    super().__init__()
    self.column_names = []
          
  def AddMatrix(self, filename):
    #filename format: headache  {"WN_noun.cognition": 0.5, "WN_noun.state": 0.5}
    for line in open(filename):
      word, json_line = line.strip().split("\t")
      self.vocab.add(word)
      json_features = json.loads(json_line)
      features = {}  
      if word in self.matrix: 
        features = self.matrix[word]
      for feature_name, feature_val in json_features.items():
        if feature_name in self.column_names:
          column_num = self.column_names.index(feature_name)
        else:
          column_num = len(self.column_names)
          self.column_names.append(feature_name)
          self.number_of_columns += 1
          print("  Added new Sense:", feature_name, "at index", column_num )
        features[column_num] = feature_val
      self.matrix[word] = features

      
class VectorMatrix(Matrix):
  def AddMatrix(self, filename):
    #filename format: biennials -0.11809 0.089522 -0.026722 0.075579 -0.02453
    binary_file = False
    if filename.endswith(".gz"):
      filename = gzip.open(filename, "rb")
      binary_file = True
    else:
      filename = open(filename)
    for line in filename:
      tokens = line.strip().split()
      word = tokens[0]
      if binary_file: 
        word = word.decode("utf-8")
      self.vocab.add(word)
      features = {}
      for dim, val in enumerate(tokens[1:]):
        features[dim] = float(val) + 1e-6
      self.matrix[word] = features
      self.number_of_columns = len(tokens)-1
      

def Similarity(v1, v2, metric="correlation"):
  def IsZero(v):
    return all(n == 0 for n in v)    

  if metric == "correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return pearsonr(v1, v2)[0]

  if metric == "abs_correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return abs(pearsonr(v1, v2)[0])

  if metric == "cosine":
    return 1 - spatial.distance.cosine(v1, v2)
    
  if metric == "heuristic1":
    result = 0 
    for i, j in zip(v1, v2):
      result += abs(i-j)
    return 1 - result
  
def Threshold(similarity_matrix, sample_size=0.75, num_resamples=10000, confidence_interval=0.95):
  #TODO
  """
  def sample_matrix():
    matrix = {}
    
  def get_threshold(matrix):
    return 0.0
    
  threshold = 0.0
  thresholds = []
  for i in range(num_resamples):
    sample = sample_matrix()
    local_threshold = get_threshold(sample)
    thresholds.append(local_threshold)
  thresholds.sort()
  if args.verbose:
    print("THRESHOLD:", threshold)
  return threshold
  """
  pass
  
def SimilarityMatrix(vsm_matrix, oracle_matrix, distance_metric="correlation", 
                    optimization_direction=GRB.MAXIMIZE):
  similarity_matrix = {}
  vocabulary = vsm_matrix.vocab & oracle_matrix.vocab
  for i in range(vsm_matrix.number_of_columns):
    for j in range(oracle_matrix.number_of_columns):      
      similarity_matrix[i,j] = Similarity(vsm_matrix.Column(i, vocabulary), 
               oracle_matrix.Column(j, vocabulary), distance_metric)      

  threshold = args.distance_metric_threshold
  if args.auto_threshold:
    threshold = Threshold(similarity_matrix)  
    
  for i in range(vsm_matrix.number_of_columns):
    for j in range(oracle_matrix.number_of_columns):      
      if similarity_matrix[i,j] < threshold:
        similarity_matrix[i,j] = 0.0

  return similarity_matrix

class ILP(object):
  """
   This class formulates and solves the following simple ILP model:

    maximize (or minimize)
          DiSj * Xij 
          -- DiSj is a correlation/absolute correlation/cosine/heuristic similarity 
             between a VSM dimension Di and a Oracle matrix dimension Sj; 
          -- Xij is an ILP (binary) variable; Xij==1 iff Di aligned to Sj
          
    subject to
          For all i sum(Xj) <= 1 
          -- only one dimension in the Oracle matrix alignes to one or many dimensions in the VSM matrix

    Xij binary
  """
  def CreateModel(self, vsm_matrix, oracle_matrix, similarity_matrix, optimization_direction):
    try:
      # Create a new model
      self.model = Model("ilp")
  
      # Create VSM dim * Oracle dim variables
      objective = LinExpr()
      for i in range(vsm_matrix.number_of_columns):
        for j in range(oracle_matrix.number_of_columns):
          # alignment variables Xij: Xij==1 iff Di aligned to Sj
          Xij_str = "X_"+str(i)+"_"+str(j) 
          Xij = self.model.addVar(vtype=GRB.BINARY, name=Xij_str) 
          Dij = similarity_matrix[i,j]
          if args.verbose:
            print("METRIC:{}\t{}\t{}".format(args.distance_metric, Xij_str, Dij))
          objective += Dij * Xij

      print("Variables created")
      # Integrate new variables
      self.model.update()
      print("Model updated")

      # Set objective
      self.model.setObjective(objective, optimization_direction)
      
      # Add constraints: 
      #"""
      # For all i sum(Xj) <= 1 (enforce many-to-one_supersense alignment)
      for i in range(vsm_matrix.number_of_columns):
        Xj = LinExpr()
        for j in range(oracle_matrix.number_of_columns):
          Xj += self.model.getVarByName("X_"+str(i)+"_"+str(j) )
        self.model.addConstr(Xj <= 1)
      #"""
      # For all j sum(Xi) <= 5 (at most 5 vector dimensions are aligned to one supersense)
      #"""
      for j in range(oracle_matrix.number_of_columns):
        Xi = LinExpr()
        for i in range(vsm_matrix.number_of_columns):
          Xi += self.model.getVarByName("X_"+str(i)+"_"+str(j))
        self.model.addConstr(Xi <= 10)
      #"""  
    except GurobiError:
      print('Gurobi Error', GurobiError.value)
      sys.exit(1)
 
def main():
  oracle_matrix = OracleMatrix()
  for filename in args.interpretations: #.split():
    print("Loading oracle matrix:", filename)
    oracle_matrix.AddMatrix(filename)
    
  vsm_matrix = VectorMatrix()
  print("Loading VSM file:", args.in_vectors)
  vsm_matrix.AddMatrix(args.in_vectors)
  #Debug(oracle_matrix, 7, vsm_matrix, 3)  

  start = timeit.timeit()
  ilp = ILP()
  distance_metric = args.distance_metric
  if args.optimization_direction == "MAXIMIZE":
    optimization_direction = GRB.MAXIMIZE  
  else: 
    optimization_direction = GRB.MINIMIZE

  similarity_matrix = SimilarityMatrix(vsm_matrix, oracle_matrix, 
                     distance_metric=distance_metric, 
                     optimization_direction=optimization_direction)
  ilp.CreateModel(vsm_matrix, oracle_matrix, similarity_matrix, optimization_direction)

  ilp.model.optimize()
  if args.verbose:
    print("DISTANCE METRIC:", distance_metric)
    print("OPTIMIZATION DIRECTION:", args.optimization_direction)
    print("VOCABULARY:")
    for w in sorted(vsm_matrix.vocab & oracle_matrix.vocab):
      print("    ", w)
    print("\n\n")
    print("OPTIMAL SOLUTION:")
    ilp.model.printAttr("X")
    
  if ilp.model.status == GRB.status.INF_OR_UNBD:
    # Turn presolve off to determine whether model is infeasible
    # or unbounded
    ilp.model.setParam(GRB.param.presolve, 0)
    ilp.model.optimize()
  if ilp.model.status == GRB.status.OPTIMAL:
    print("Optimal objective: %g" % ilp.model.objVal)
    ilp.model.write(args.out_file + ".sol")
    end = timeit.timeit()
    print("Computation time: ", end - start)
    exit(0)
  elif ilp.model.status != GRB.status.INFEASIBLE:
    print("Optimization was stopped with status %d" % ilp.model.status)
    print("Computation time: ", end - start)
    exit(0)
  # Model is infeasible - compute an Irreducible Infeasible Subsystem (IIS)
  print("")
  print("Model is infeasible")
  ilp.model.computeIIS()
  print("IIS written to file 'model.ilp'")
  ilp.model.write("model.ilp")
  print("Computation time: ", end - start)
  
  
def Debug(oracle_matrix, col_oracle, vsm_matrix, col_vsm):
  print("**********ORACLE************")
  print(oracle_matrix)
  print("**********VSM************")
  print(vsm_matrix)
  print("****************************")
  print("Oracle:", oracle_matrix.column_names[col_oracle])
  oracle_col = oracle_matrix.Column(col_oracle, set(["cat", "dog", "ice"]))
  print("Supersense column:", oracle_col)
  vsm_col = vsm_matrix.Column(col_vsm, set(["cat", "dog", "ice"]))
  print("VSM column:", vsm_col)
  print("Similarity:", Similarity(vsm_col, oracle_col, "heuristic1"))

if __name__ == '__main__':
  main()
