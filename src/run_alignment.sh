#!/bin/bash

function get_alignments {
  VECTORS=$1
  ORACLE_MATRIX=$2
  POS=$3
  DISTANCE_METRIC=$4
  LAMBDA=$5
  OPTIMIZATION_DIRECTION=$6
  
  OUT_DIR=../dev/`basename ${VECTORS}`
  OUT_FILE=${OUT_DIR}/`basename ${ORACLE_MATRIX}`"-"${DISTANCE_METRIC}
  
  mkdir -p ${OUT_DIR}
  nice -n 2 ./interpret.py --in_vectors ${VECTORS} \
      --interpretations ${ORACLE_MATRIX} \
      --out_file ${OUT_FILE} \
      --distance_metric ${DISTANCE_METRIC} \
      --regularization_lambda ${LAMBDA} \
      --optimization_direction ${OPTIMIZATION_DIRECTION} \
      --verbose --tune_lambda 2>&1 | tee ${OUT_FILE}".log" #&
}

LAMBDA=0.0

VECTORS=/usr0/home/ytsvetko/usr3/ivsm/vectors
ORACLES=/usr0/home/ytsvetko/projects/ivsm/data/oracles
for model_name in ${VECTORS}/*.txt ; do
  echo ${model_name}
  for oracle_matrix in ${ORACLES}/* ; do
    pos="noun_verb"
    get_alignments ${model_name} ${oracle_matrix} ${pos} abs_correlation ${LAMBDA} MAXIMIZE
  done
done

