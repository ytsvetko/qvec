#!/usr/bin/env python

"""
./supersense_matrix.py --out_file  <filename>

./supersense_matrix.py --word_counter_threshold 5 --out_file oracles/semcor_noun_verb.supersenses

"""
from __future__ import division
import argparse
import json
import collections
from nltk.corpus import semcor, wordnet as wn


parser = argparse.ArgumentParser()
parser.add_argument("--word_counter_threshold", type=int, default=0, help="minimal number of occurrences in corpus")
parser.add_argument("--out_file", default="oracles/semcor_noun_verb.supersenses")
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
      for k, v in self.supersenses.iteritems():
        self.supersenses[k] = v/self.counter
      self.normalized = True
    
  def __repr__(self):
    return json.dumps(self.supersenses)
    
def CollectSemcorSupersenses():
  oracle_matrix = collections.defaultdict(WordSupersenses)
  for sent in semcor.tagged_sents(tag='both'):
    for chk in sent:
      if chk.node and len(chk.node)>3 and chk.node[-3]=='.' and chk.node[-2:].isdigit():
        if chk[0].node.startswith('N'):
          pos = "n"
        elif chk[0].node.startswith('V'):
          pos = "v"
        else:
          continue
        lemmas = chk.node[:-3]
        wnsn = int(chk.node[-2:])
        ssets = wn.synsets(lemmas, pos)
        sorted_ssets = sorted(ssets, key=lambda x: x.name)
        filtered_ssets = None
        for lemma in lemmas.split("_"):  
          if not filtered_ssets or len(filtered_ssets) == 0:
            filtered_ssets = filter(lambda x: lemma in x.name, sorted_ssets)
        if filtered_ssets and len(filtered_ssets) > 0:
          sorted_ssets = filtered_ssets
        try:
          supersense = sorted_ssets[wnsn-1].lexname # prints 'noun.group
        except:
          #print("."),
          continue
        for lemma in lemmas.split("_"):        
          ssets = wn.synsets(lemma, pos)
          if len(ssets) > 0:
            if lemma.isdigit():
              lemma = "0"
            oracle_matrix[lemma].Add(supersense, "semcor")  
  return oracle_matrix      
 
def main():
  oracle_matrix = CollectSemcorSupersenses()
  out_f = open(args.out_file, "w")
  for lemma, supersenses in sorted(oracle_matrix.iteritems()):
    if supersenses.counter < args.word_counter_threshold:
      continue
    supersenses.NormalizeProbs()
    out_f.write("{}\t{}\n".format(lemma, str(supersenses)))

if __name__ == '__main__':
  main()
