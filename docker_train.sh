#! /bin/bash

docker run --gpus all --name hrnet --ipc=host -it --rm -v $PWD:/tmp/hrnet \
    hrnet \
    bash -c 'cd hrnet; \
    python3 -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    --cfg experiments/cityscapes/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
    OUTPUT_DIR output1.6.0'

