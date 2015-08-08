#!/usr/bin/env python

"""
./ptb_matrix.py --out_file  <filename>

./ptb_matrix.py --word_counter_threshold 5 --out_file oracles/ptb_noun_verb.pos_tags.filtered

./ptb_matrix.py --word_counter_threshold 5 --out_file oracles/ptb_noun_verb.pos_tags.filtered | sort - >  ../vectors/wang2vec/classes.txt

"""
from __future__ import division
import argparse
import json
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--word_counter_threshold", type=int, default=0, help="minimal number of occurrences in corpus")
parser.add_argument("--out_file", default="oracles/ptb_noun_verb.pos_tags")
parser.add_argument("--PTB", default="/mal2/corpora/penn_tb_3.0_preprocessed/train.1.notraces")
args = parser.parse_args()

class WordTags(object):
  def __init__(self):
    self.counter = 0
    self.normalized = False
    self.tags = collections.defaultdict(int)
  
  def Add(self, tag, source=None):
    if source:
      tag = source + "." + tag  
    self.tags[tag] += 1
    self.counter += 1
    #TODO handle self.normalized
    
  def NormalizeProbs(self):
    if not self.normalized:
      for k, v in self.tags.iteritems():
        self.tags[k] = v/self.counter
      self.normalized = True
    
  def __repr__(self):
    return json.dumps(self.tags)
    
def CollectPOStags():
  oracle_matrix = collections.defaultdict(WordTags)
  for sent in open(args.PTB):
    tokens = sent.split()
    for i, token in enumerate(tokens):
      if i == 0:
        continue
      #if (tokens[i].endswith(")") and tokens[i-1].startswith("(")): #
      if (tokens[i].endswith(")") and (tokens[i-1].startswith("(N") or tokens[i-1].startswith("(V"))):
        word = tokens[i].replace(")", "").lower()
        if word.isdigit():
          word = "0"
        #pos = tokens[i-1].replace("(", "").lower()
        pos = tokens[i-1][1].lower() 
        oracle_matrix[word].Add(pos, "ptb")  
  return oracle_matrix      
 
def main():
  oracle_matrix = CollectPOStags()
  out_f = open(args.out_file, "w")
  for lemma, tags in sorted(oracle_matrix.iteritems()):
    if tags.counter < args.word_counter_threshold:
      continue
    tags.NormalizeProbs()
    out_f.write("{}\t{}\n".format(lemma, str(tags)))
    if len(tags.tags) == 1:
      print("{} {}".format(tags.tags.keys()[0][-1].upper(), lemma))

if __name__ == '__main__':
  main()
