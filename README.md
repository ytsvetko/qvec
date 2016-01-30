QVEC
========
Yulia Tsvetkov, ytsvetko@cs.cmu.edu

This is an easy-to-use, fast tool to measure the intrinsic quality of word vectors. The
evaluation score depends on how well the word vectors correlate with a matrix of features
from manually crafted lexical resources. The evaluation score is shown to correlate strongly
with performance in downstream tasks (cf. Tsvetkov et al, 2015 for details and results). QVEC
is model agnostic and thus can be used for evaluating word vectors produced by any
given model.

<a href="http://www.cs.cmu.edu/~ytsvetko/papers/qvec.pdf">Evaluation of Word Vector Representations by Subspace Alignment</a>
  </li> 

### Usage

Each vector file should have one word vector per line as follows (space delimited):-

```the -1.0 2.4 -0.3 ...```

#### Semantic content evaluation: 

```py
./qvec_cca.py --in_vectors  ${your_vectors} --in_oracle  oracles/semcor_noun_verb.supersenses.en    
```
To obtain vector column labels, add the --interpret parameter; to print top K values in each dimension add --top K: 

```py
./qvec.py --in_vectors ${your_vectors} --in_oracle oracles/semcor_noun_verb.supersenses.en --interpret --top 10
```

#### Multilingual evaluation for English, Danish, and Italian: 

```py
./qvec_cca.py --in_vectors  ${your_vectors} --in_oracle   --in_oracle oracles/semcor_noun_verb.supersenses.en,oracles/semcor_noun_verb.supersenses.it,oracles/semcor_noun_verb.supersenses.da 
```

#### Syntactic content evaluation: 

```py
./qvec_cca.py --in_vectors  ${your_vectors} --in_oracle  oracles/ptb.pos_tags    
```


### Citation:
    @InProceedings{qvec:enmlp:15,
    author = {Tsvetkov, Yulia and Faruqui, Manaal and Ling, Wang and Lample, Guillaume and Dyer, Chris},
    title={Evaluation of Word Vector Representations by Subspace Alignment},
    booktitle={Proc. of EMNLP},
    year={2015},
    }

This repository is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/

