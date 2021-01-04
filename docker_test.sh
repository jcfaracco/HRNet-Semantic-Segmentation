#! /bin/bash

docker run --gpus all --name hrnet --ipc=host -it --rm -v $PWD:/tmp/hrnet \
    hrnet \
    bash -c 'cd hrnet; \
    python3 tools/test.py \
    --cfg experiments/cityscapes/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
    TEST.MODEL_FILE output/cityscapes/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/final_state.pth \
    TEST.BATCH_SIZE_PER_GPU 32'

