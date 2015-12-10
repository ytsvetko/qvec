#!/usr/bin/env python3

"""
./supersense_matrix.py --out_file  <filename>

./supersense_matrix.py --word_counter_threshold 5 --out_file oracles/semcor_noun_verb.supersenses.da

"""
import argparse
import json
import collections
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--word_counter_threshold", type=int, default=0, help="minimal number of occurrences in corpus")
parser.add_argument("--annotations", default="../../multi-supersense/italian/evalita2011_supersense_utf8.txt")
parser.add_argument("--out_file", default="../oracles/semcor_noun_verb.supersenses.it")
args = parser.parse_args()

class WordSupersenses(object):
  def __init__(self):
    self.counter = 0
    self.normalized = False
    self.supersenses = collections.defaultdict(int)
  
  def Add(self, supersense, source=None):
    if source:
      supersense = source + "." + supersense  
    self.supersenses[supersense] += 1
    self.counter += 1
    #TODO handle self.normalized
    
  def NormalizeProbs(self):
    if not self.normalized:
      for k, v in self.supersenses.items():
        self.supersenses[k] = v/self.counter
      self.normalized = True
    
  def __repr__(self):
    return json.dumps(self.supersenses)
    
def CollectSemcorSupersenses(annotations):
  oracle_matrix = collections.defaultdict(WordSupersenses)
  for line in open(annotations):
    if not line.strip():
      continue
    # 10      Carstens        carstens        PROPN   B-noun.building
    tokens = line.split()
    assert len(tokens) == 4, line
    lemma = tokens[0].lower()
    if lemma.isdigit():
      lemma = "0"
    elif not lemma.isalpha():
      continue
    if "-" not in tokens[-1]:
      continue
    supersense = tokens[-1].split("-")[-1]
    if supersense.startswith("ad"):
      continue
    oracle_matrix[lemma].Add(supersense, "semcor")  
  return oracle_matrix      
 
def main():
  oracle_matrix = CollectSemcorSupersenses(args.annotations)
  out_f = open(args.out_file, "w")
  for lemma, supersenses in sorted(oracle_matrix.items()):
    if supersenses.counter < args.word_counter_threshold:
      continue
    supersenses.NormalizeProbs()
    out_f.write("{}\t{}\n".format(lemma, str(supersenses)))

if __name__ == '__main__':
  main()
