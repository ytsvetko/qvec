#!/usr/bin/env python3

"""
./qvec_cca.py --in_vectors <filename> --in_oracle <filename1,filename2..,filenameN> --verbose

"""

import argparse
import json
import gzip
import sys
import subprocess
import os
import numpy as np
import scipy
from scipy import linalg
from scipy.linalg import decomp_qr
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--in_vectors", default="vectors/w2v_sg_1b_100.txt")
parser.add_argument("--in_oracle", default="oracles/semcor_noun_verb.supersenses.en", help="comma-separated list of linguistic annotation files, each is in format word \\t json dictionary of linguistic features")
parser.add_argument("--verbose", action='store_true')
args = parser.parse_args()


def GetVocab(file_list, vocab_union=False):
  def file_vocab(filename):
    vocab = set()
    binary_file = False
    if filename.endswith(".gz"):
      f = gzip.open(filename, "rb")
      binary_file = True
    else:
      f = open(filename)
    for line in f:
      tokens = line.split()
      if len(tokens) <= 2: #ignore w2v first line
        continue
      word = tokens[0]
      if binary_file: 
        word = word.decode("utf-8")
      vocab.add(word)
    return vocab
  vocab = set()
  for f in file_list:
    vocab_f = file_vocab(f)
    if not vocab:
      vocab = vocab_f
    else:
      if vocab_union:
        vocab = vocab | vocab_f
      else: #intersection
        vocab = vocab & vocab_f 
  return sorted(vocab)

def ReadOracleMatrix(filenames, vocab_set):
  column_names = set()
  matrix = {}
  for filename in filenames:
    # file format: headache  {"WN_noun.cognition": 0.5, "WN_noun.state": 0.5}
    if args.verbose:
      print("Loading oracle matrix:", filename)    

    for line in open(filename):
      word, json_line = line.strip().split("\t")
      if word not in vocab_set:
        continue
      features = json.loads(json_line)
      column_names.update(features.keys())     
      if word not in matrix:
        matrix[word] = features
      else:
        prev_features = matrix[word]
        matrix[word] = combine_dicts(features, prev_features)

  if args.verbose:
   print("Number of oracle features:", len(column_names))

  column_name_dict = {feature_name:index for index, feature_name in enumerate(sorted(column_names))}

  result = np.zeros((len(vocab_set), len(column_names)))
  for row_num, word in enumerate(sorted(vocab_set)):
    for feature_name, feature_val in matrix[word].items():
      result[row_num, column_name_dict[feature_name]] = feature_val
  return result
  
def combine_dicts(A, B):
  return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}
        
def ReadVectorMatrix(filename, vocab_set):
  #filename format: biennials -0.11809 0.089522 -0.026722 0.075579 -0.02453
  binary_file = False
  if filename.endswith(".gz"):
    f = gzip.open(filename, "rb")
    binary_file = True
  else:
    f = open(filename)

  matrix = {}
  for line in f:
    tokens = line.strip().split()
    if len(tokens) <= 2: #ignore w2v first line
      continue
    word = tokens[0]
    if binary_file: 
      word = word.decode("utf-8")
    if word not in vocab_set:
      continue

    number_of_columns = len(tokens)-1

    features = []
    for dim, val in enumerate(tokens[1:]):
      val = float(val)
      features.append(val)
    matrix[word] = features
    
  result = []
  for word in sorted(vocab_set):
    assert word in matrix, word
    result.append(matrix[word])
  return np.array(result)

def NormCenterMatrix(M):
  M = preprocessing.normalize(M)
  m_mean = M.mean(axis=0)
  M -= m_mean
  return M

def ComputeCCA(X, Y):
  assert X.shape[0] == Y.shape[0], (X.shape, Y.shape, "Unequal number of rows")
  assert X.shape[0] > 1, (X.shape, "Must have more than 1 row")
  
  X = NormCenterMatrix(X)
  Y = NormCenterMatrix(Y)
  X_q, _, _ = decomp_qr.qr(X, overwrite_a=True, mode='economic', pivoting=True)
  Y_q, _, _ = decomp_qr.qr(Y, overwrite_a=True, mode='economic', pivoting=True)
  C = np.dot(X_q.T, Y_q)
  r = linalg.svd(C, full_matrices=False, compute_uv=False)
  d = min(X.shape[1], Y.shape[1])
  r = r[:d]
  r = np.minimum(np.maximum(r, 0.0), 1.0)  # remove roundoff errs
  return r.mean()

def main():
  oracle_files = args.in_oracle.strip().split(",")
  vocab_oracle = GetVocab(oracle_files, vocab_union=True)
  vocab_vectors = GetVocab([args.in_vectors])
  vocab_set = set(vocab_vectors) & set(vocab_oracle)
  if len(vocab_set) < 1000:
    print("*** Warning: vocabulary size is too small. ***")
  if args.verbose:
    print("Vocabulary size:", len(vocab_set))
  
  oracle_matrix = ReadOracleMatrix(oracle_files, vocab_set)

  if args.verbose:
    print("Loading VSM file:", args.in_vectors)
  vsm_matrix = ReadVectorMatrix(args.in_vectors, vocab_set)
  
  if args.verbose:
    print("Running CCA")
  cca_result = ComputeCCA(vsm_matrix, oracle_matrix)
  print("QVEC_CCA score: %g" % cca_result)
  
if __name__ == '__main__':
  main()
