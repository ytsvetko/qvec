#!/bin/bash


function get_alignments {
  MODEL_NUM=$1
  ORACLE_MATRIX=$2
  POS=$3
  DISTANCE_METRIC=$4
  OPTIMIZATION_DIRECTION=$5
  
  MODEL_DIR=../data/${MODEL_NUM}"thModel"
  OUT_DIR=../work/${MODEL_NUM}"thModel"/${POS}"-"${DISTANCE_METRIC}"-"${OPTIMIZATION_DIRECTION}

  mkdir -p ${OUT_DIR}
  
  for f in ${MODEL_DIR}/*.txt ; do
    echo `basename $f`    
    echo ${OUT_DIR}/`basename $f`
    nice -n 2 ./interpret.py --in_vectors $f \
      --interpretations ${ORACLE_MATRIX} \
      --out_file ${OUT_DIR}/`basename $f` \
      --distance_metric ${DISTANCE_METRIC} \
      --optimization_direction ${OPTIMIZATION_DIRECTION} \
      --verbose > ${OUT_DIR}/`basename $f`".log" &
  done  
}

MODEL_NUM=( 5 8 )
POS=( "noun" "verb" "adj" )

for model_num in "${MODEL_NUM[@]}" ; do
  for pos in "${POS[@]}" ; do
    oracle_matrix="../data/supersenses/wn_"${pos}".supersneses"
    get_alignments ${model_num} ${oracle_matrix} ${pos} correlation MAXIMIZE
    get_alignments ${model_num} ${oracle_matrix} ${pos} abs_correlation MAXIMIZE
    get_alignments ${model_num} ${oracle_matrix} ${pos} cosine MINIMIZE
    get_alignments ${model_num} ${oracle_matrix} ${pos} heuristic1 MINIMIZE
  done
done

