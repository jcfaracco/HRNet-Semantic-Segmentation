#! /bin/bash

CFG=$1
SUFFIX=$2

NAME=`basename $CFG`
NAME="${NAME%.*}"

# docker run --gpus all --name hrnet --ipc=host -it --rm \
docker run --gpus '"device=4,5,6,7"' --name hrnet2 --ipc=host -it --rm \
   -v $PWD:/tmp/hrnet \
   -v /data/home/jordao/MO434/TACO:/TACO \
   hrnet \
   bash -c "cd hrnet; \
   python3 -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
   --cfg $CFG \
   OUTPUT_DIR results/${NAME}_${SUFFIX}"

