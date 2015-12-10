#!/usr/bin/env python2.7

"""
./qvec.py --in_vectors <filename> --in_oracle <filename> --interpret

"""
from __future__ import absolute_import
import argparse
import json
import numpy as np
from scipy.stats.stats import pearsonr
from scipy import spatial
import time
import gzip
import sys
from io import open

parser = argparse.ArgumentParser()
parser.add_argument("--in_vectors", default="vectors/w2v_sg_1b_100.txt")
parser.add_argument("--in_oracle", default="oracles/semcor_noun_verb.supersenses.en", help="comma-separated list of linguistic annotation files, each is in format word \\t json dictionary of linguistic features")
parser.add_argument("--distance_metric", default="correlation",
                    help="correlation, abs_correlation, cosine")
parser.add_argument("--interpret", action='store_true')
parser.add_argument("--top", type=int, default=100)
parser.add_argument("--verbose", action='store_true')
args = parser.parse_args()


class TopK(object):
  def __init__(self, k):
    self.k = k
    self.elements = []
    self.sorted = False

  def Push(self, word, value):
    if len(self.elements) < self.k:
      self.elements.append((word, value))
    else:
      self.Sort()
      if self.elements[-1][1] < value:
        self.elements[-1] = (word, value)
        self.sorted = False
  
  def Sort(self):
    if not self.sorted:
      self.elements = sorted(self.elements, key=lambda x: x[1], reverse=True)
      self.sorted = True

  def GetSortedElements(self):
    self.Sort()
    return self.elements


class Matrix(object):
  def __init__(self):
    self.matrix = {} #key - word; value = dict with key - column num, value - val
    self.number_of_columns = 0

  def Column(self, dim, vocab):
    #return dimension based on vocab
    column = []
    for word in sorted(vocab):
      column.append(self.matrix[word].get(dim, 0.0))
    return column

  def __repr__(self):
    result = []
    for word in sorted(self.matrix):
      line = [word]
      features = self.matrix[word]
      for col in xrange(self.number_of_columns):
        line.append(unicode(features.get(col, 0.0)))
      result.append(" ".join(line))
    return "\n".join(result)


class OracleMatrix(Matrix):
  def __init__(self):
    super(OracleMatrix, self).__init__()
    self.column_names = []
          
  def AddMatrix(self, filename):
    #filename format: headache  {"WN_noun.cognition": 0.5, "WN_noun.state": 0.5}
    for line in open(filename):
      word, json_line = line.strip().split("\t")
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
          if args.verbose:
            print "  Added new oracle column:", feature_name, "at index", column_num
        features[column_num] = feature_val
      self.matrix[word] = features


class VectorMatrix(Matrix):
  def AddMatrix(self, filename, top_k=0):
    #filename format: biennials -0.11809 0.089522 -0.026722 0.075579 -0.02453
    binary_file = False
    if filename.endswith(".gz"):
      f = gzip.open(filename, "rb")
      binary_file = True
    else:
      f = open(filename)
    self.best_in_column = []
    for line in f:
      tokens = line.strip().split()
      if len(tokens) == 2: #ignore w2v first line
        continue
      word = tokens[0]
      if binary_file: 
        word = word.decode("utf-8")

      self.number_of_columns = len(tokens)-1
      if top_k and len(self.best_in_column) == 0:
        self.best_in_column = [TopK(top_k) for _ in xrange(self.number_of_columns)]

      features = {}
      for dim, val in enumerate(tokens[1:]):
        val = float(val)
        features[dim] = val
        if top_k:
          self.best_in_column[dim].Push(word, val)

      self.matrix[word] = features


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
    return spatial.distance.cosine(v1, v2)

def SimilarityMatrix(vsm_matrix, oracle_matrix, distance_metric="correlation"):
  similarity_matrix = np.zeros((vsm_matrix.number_of_columns, oracle_matrix.number_of_columns))
  vocabulary = set(vsm_matrix.matrix.keys()) & set(oracle_matrix.matrix.keys())
  for i in xrange(vsm_matrix.number_of_columns):
    for j in xrange(oracle_matrix.number_of_columns):      
      similarity_matrix[i,j] = Similarity(vsm_matrix.Column(i, vocabulary), 
          oracle_matrix.Column(j, vocabulary), distance_metric)

  return similarity_matrix

def AlignColumns(vsm_matrix, oracle_matrix, distance_metric):
  similarity_matrix = SimilarityMatrix(vsm_matrix, oracle_matrix, 
                                       distance_metric=distance_metric)
  total_score = 0
  alignments = []
  for i in xrange(vsm_matrix.number_of_columns):
    best_oracle_column = np.argmax(similarity_matrix[i])
    similarity = similarity_matrix[i, best_oracle_column]
    alignments.append((best_oracle_column, similarity))
    total_score += similarity
  return alignments, total_score

def main():
  start = time.time()

  distance_metric = args.distance_metric

  oracle_matrix = OracleMatrix()
  for filename in args.in_oracle.strip().split(","):
    if args.verbose:
      print "Loading oracle matrix:", filename 
    oracle_matrix.AddMatrix(filename)

  vsm_matrix = VectorMatrix()
  if args.verbose:
    print "Loading VSM file:", args.in_vectors
  top_k = args.top if args.interpret else 0
  vsm_matrix.AddMatrix(args.in_vectors, top_k)

  alignments, score = AlignColumns(vsm_matrix, oracle_matrix, distance_metric)

  print "QVEC score: ", score
  if args.interpret:
    print "\t".join(["Dimension", "Aligned_oracle_column", "Similarity", "Top-N_words"])
    for i in xrange(vsm_matrix.number_of_columns):
      top_words = []
      if vsm_matrix.best_in_column:
        top_words = [word for (word, value) in vsm_matrix.best_in_column[i].GetSortedElements()]
      print "{}\t{}\t{}\t{}".format(
          i,
          oracle_matrix.column_names[alignments[i][0]],
          alignments[i][1],
          " ".join(top_words))
  print "Computation time: ", time.time() - start

if __name__ == u'__main__':
  main()
