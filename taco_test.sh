#! /bin/bash

docker run --gpus all --name hrnet --ipc=host -it --rm \
    -v $PWD:/tmp/hrnet \
    -v /data/home/jordao/MO434/TACO:/TACO \
    hrnet \
    bash -c 'cd hrnet; \
    python3 tools/test.py \
    --cfg experiments/taco/seg_hrnet_w48_cls60_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml \
    DATASET.TEST_SET testval \
    TEST.MODEL_FILE results/output1.6.0/taco/seg_hrnet_w48_cls60_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200/final_state.pth \
    OUTPUT_DIR results/tacotest'

