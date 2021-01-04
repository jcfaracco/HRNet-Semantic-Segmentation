#! /bin/bash

docker run --gpus '"device=4,5,6,7"' --name fcn --ipc=host -it --rm \
    -v $PWD:/tmp/hrnet \
    -v /data/home/jordao/MO434/TACO:/TACO \
    hrnet \
    bash -c 'cd hrnet; \
    python3 -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    --cfg experiments/taco/fcn_cls5_720x720_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml \
    OUTPUT_DIR results/fcn'

