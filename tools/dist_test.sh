#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-$((RANDOM % 10000 + 10000))}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
