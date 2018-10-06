#
# Evaluation for eager system
#
# It will convert actions into AMR graph and call smatch
# to evaluate it score
# 
# Usage:
#
#   bash eval_eager.sh predict-action gold-AMR
#
#!/bin/bash
BASEDIR=$(dirname "$0")/../../amr_aligner
python ${BASEDIR}/eager_actions_evaluator.py -pred_actions $1 -gold $2
