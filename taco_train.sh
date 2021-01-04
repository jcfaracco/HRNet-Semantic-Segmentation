#! /bin/bash

docker run --gpus all --name hrnet --ipc=host -it --rm \
    -v $PWD:/tmp/hrnet \
    -v /data/home/jordao/MO434/TACO:/TACO \
    hrnet \
    bash -c 'cd hrnet; \
    python3 -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    --cfg experiments/taco/seg_hrnet_w48_cls11_720x720_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml \
    OUTPUT_DIR results/class10'

