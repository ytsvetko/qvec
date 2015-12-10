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
parser.add_argument("--annotations", default="../../multi-supersense/danish/semdax/supersenses/official_distribution_pos_and_lemma/")
parser.add_argument("--annotation_mapping", default="../../multi-supersense/danish/semdax/supersenses/resources/map_to_common_supersenses.tsv")
parser.add_argument("--out_file", default="../oracles/semcor_noun_verb.supersenses.da")
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
    
def ReadSupersenseMapping(filename):
  mapping={} # key: Danish supersense, value: English supersense
  for line in open(filename):
    tokens = line.split()
    if not tokens[0] == "da" or "-" not in tokens[2]:
      continue
    mapping[tokens[1].split("-")[-1]] = tokens[2].split("-")[-1]
  return mapping
  
def CollectSemcorSupersenses(annotations, mapping):
  oracle_matrix = collections.defaultdict(WordSupersenses)
  for root, _, files in os.walk(annotations):
    for f in files:
      if not f.endswith((".poslemma")):
        continue
      for line in open(os.path.join(root, f)):
        if not line.strip():
          continue
        # 10      Carstens        carstens        PROPN   B-noun.building
        tokens = line.split()
        assert len(tokens) == 5, line
        lemma = tokens[1].lower()
        if lemma.isdigit():
          lemma = "0"
        elif not lemma.isalpha():
          continue
        if "-" not in tokens[-1]:
          continue
        supersense = tokens[-1].split("-")[-1]
        if supersense.startswith("ad"):
          continue
        if supersense in mapping:
          oracle_matrix[lemma].Add(mapping[supersense], "semcor")  
  return oracle_matrix      
 
def main():
  mapping = ReadSupersenseMapping(args.annotation_mapping)
  oracle_matrix = CollectSemcorSupersenses(args.annotations, mapping)
  out_f = open(args.out_file, "w")
  for lemma, supersenses in sorted(oracle_matrix.items()):
    if supersenses.counter < args.word_counter_threshold:
      continue
    supersenses.NormalizeProbs()
    out_f.write("{}\t{}\n".format(lemma, str(supersenses)))

if __name__ == '__main__':
  main()
