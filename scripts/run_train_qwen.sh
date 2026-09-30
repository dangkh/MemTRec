#!/usr/bin/env bash
# Run MemRec evaluation with an open-source model (Qwen2.5-7B-Instruct,
# loaded in-process via HuggingFace transformers - no API key/server needed),
# on a dataset converted from AgenticRec_CFmemory by convert_agenticrec_to_memrec.py.
#
# This is the open-source counterpart to the Azure OpenAI-based
# configs/memrec_<dataset>.yaml - same data, same fixed candidate set,
# different LLM provider (see configs/memrec_<dataset>_qwen.yaml,
# provider.name: local_hf).
#
# Usage:
#   bash scripts/run_train_qwen.sh <dataset> [--number_of_users N] [extra run_train.py args...]
#
# --number_of_users N picks the SAME first-N users (same order) as
# AgenticRec_CFmemory's own agent_rec_*.py scripts use for --number_of_users N
# (see scripts/generate_eval_user_list.py) - auto-generated here if missing.
# Without it, the dataset's default configs/memrec_<dataset>_qwen.yaml
# eval_user_list (first 1000 users) is used.
#
# Examples:
#   bash scripts/run_train_qwen.sh yelp
#   bash scripts/run_train_qwen.sh Video_Game --number_of_users 500 --device cuda:0
#   bash scripts/run_train_qwen.sh MIND --number_of_users 500 --device cuda:0
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_train_qwen.sh <dataset> [--number_of_users N] [extra run_train.py args...]"
    echo "  <dataset> one of: yelp, Video_Game, Books, CDs_and_Vinyl, ml, MIND"
    exit 1
fi

DATASET="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_ROOT/configs/memrec_${DATASET}_qwen.yaml"
INTER_FILE="$PROJECT_ROOT/data/processed/${DATASET}/${DATASET}.inter"
CONVERT_SCRIPT="convert_agenticrec_to_memrec.py --data_name $DATASET"
if [ "$DATASET" = "MIND" ]; then
    CONVERT_SCRIPT="convert_mind_to_memrec.py"
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: no config at $CONFIG"
    echo "Run: python scripts/$CONVERT_SCRIPT"
    exit 1
fi
if [ ! -f "$INTER_FILE" ]; then
    echo "Error: dataset not converted yet - missing $INTER_FILE"
    echo "Run: python scripts/$CONVERT_SCRIPT"
    exit 1
fi

# Pull --number_of_users N out of the args (not a run_train.py flag) and turn
# it into --eval_user_list pointing at the matching fixed user list.
EXTRA_ARGS=()
NUMBER_OF_USERS=""
while [ $# -gt 0 ]; do
    if [ "$1" = "--number_of_users" ]; then
        NUMBER_OF_USERS="$2"
        shift 2
    else
        EXTRA_ARGS+=("$1")
        shift
    fi
done

if [ -n "$NUMBER_OF_USERS" ]; then
    EVAL_USER_LIST="$PROJECT_ROOT/data/processed/${DATASET}/${DATASET}.eval_user_list_n${NUMBER_OF_USERS}.json"
    if [ ! -f "$EVAL_USER_LIST" ]; then
        echo "Generating eval_user_list for $DATASET, first $NUMBER_OF_USERS users..."
        python3 "$SCRIPT_DIR/generate_eval_user_list.py" --data_name "$DATASET" --number_of_users "$NUMBER_OF_USERS"
    fi
    EXTRA_ARGS+=(--eval_user_list "$EVAL_USER_LIST")
fi

echo "Dataset:      $DATASET"
echo "Config:       $CONFIG"
echo "LLM provider: local_hf (Qwen/Qwen2.5-7B-Instruct, open-source, no API key)"
if [ -n "$NUMBER_OF_USERS" ]; then
    echo "Eval users:   first $NUMBER_OF_USERS (matching agent_rec_*.py --number_of_users $NUMBER_OF_USERS)"
fi
echo

python3 "$SCRIPT_DIR/run_train.py" \
    --model memrec_agent \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}"
