#! /bin/bash

CFG=$1
WEIGHTS=$2
OUTDIR=$3


#    --gpus device=4,5,6,7 \
docker run \
    --gpus '"device=4,5,6,7"' \
    --name hrnet --ipc=host -it --rm \
    -v $PWD:/tmp/hrnet \
    -v /data/home/jordao/MO434/TACO:/TACO \
    hrnet \
    bash -c "cd hrnet; \
    python3 tools/test.py \
    --cfg $CFG \
    DATASET.TEST_SET test \
    TEST.MODEL_FILE $WEIGHTS \
    OUTPUT_DIR $OUTDIR"

