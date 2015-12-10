#!/usr/bin/env python3

"""
./qvec_cca.py --in_vectors <filename> --in_oracle <filename1,filename2..,filenameN> --verbose

"""

import argparse
import json
import gzip
import sys
import subprocess

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
  
def ReadOracleMatrix(filename, vocab, column_names=None, matrix=None):
  #filename format: headache  {"WN_noun.cognition": 0.5, "WN_noun.state": 0.5}
  vocab_set = set(vocab)
  if not column_names:
    column_names = []
    matrix = {}
      
  for line in open(filename):
    word, json_line = line.strip().split("\t")
    if word not in vocab_set:
      continue
    json_features = json.loads(json_line)
    features = {}  
    for feature_name, feature_val in json_features.items():
      if feature_name in column_names:
        column_num = column_names.index(feature_name)
      else:
        column_num = len(column_names)
        column_names.append(feature_name)
        if args.verbose:
          print("  Added new oracle column:", feature_name, "at index", column_num)
      features[column_num] = feature_val
      if word not in matrix:
        matrix[word] = features
      else:
        prev_features = matrix[word]
        matrix[word] = combine_dicts(features, prev_features)
  result = []
  number_of_columns = len(column_names)

  for row_num, word in enumerate(vocab):
    if word in matrix:
      row = [0] * number_of_columns
      for col_num, col_val in matrix[word].items():
        row[col_num] = col_val
      result.append(row)
  return result, column_names, matrix
  
def combine_dicts(A, B):
  return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}
        
def ReadVectorMatrix(filename, vocab):
  #filename format: biennials -0.11809 0.089522 -0.026722 0.075579 -0.02453
  binary_file = False
  if filename.endswith(".gz"):
    f = gzip.open(filename, "rb")
    binary_file = True
  else:
    f = open(filename)

  matrix = {}
  vocab_set = set(vocab)
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
  for word in vocab:
    assert word in matrix, word
    result.append(matrix[word])
  return result

def WriteMatrix(matrix, filename):
  f_out = open(filename, "w")
  for row in matrix:
    f_out.write("{}\n".format(" ".join([str(val) for val in row])))

def main():
  oracle_files = args.in_oracle.strip().split(",")
  vocab_oracle = GetVocab(oracle_files, vocab_union=True)
  vocab_vectors = GetVocab([args.in_vectors])
  vocab = sorted(set(vocab_vectors) & set(vocab_oracle))
  if len(vocab) < 1000:
    print("*** Warning: vocabulary size is too small. ***")
  if args.verbose:
    print("Vocabulary size:", len(vocab))
  
  column_names, tmp_matrix = None, None
  for filename in oracle_files:
    if args.verbose:
      print("Loading oracle matrix:", filename)
    oracle_matrix, column_names, tmp_matrix = ReadOracleMatrix(
         filename, vocab, column_names, tmp_matrix)

  if args.verbose:
    print("Loading VSM file:", args.in_vectors)
  vsm_matrix = ReadVectorMatrix(args.in_vectors, vocab)

  WriteMatrix(vsm_matrix, "X")
  WriteMatrix(oracle_matrix, "Y")

  subprocess.call(["matlab -nosplash -nodisplay -r \"cca(\'%s\',\'%s\')\"" % ("X", "Y")],shell=True)

if __name__ == '__main__':
  main()
