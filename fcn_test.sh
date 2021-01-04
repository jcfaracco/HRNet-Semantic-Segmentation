#! /bin/bash

docker run --gpus all --name fcn --ipc=host -it --rm \
    -v $PWD:/tmp/hrnet \
    -v /data/home/jordao/MO434/TACO:/TACO \
    hrnet \
    bash -c 'cd hrnet; \
    python3 tools/test.py \
    --cfg experiments/taco/fcn_cls5_720x720_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml \
    DATASET.TEST_SET testval \
    TEST.MODEL_FILE results/fcn/taco/fcn_cls5_720x720_sgd_lr4e-3_wd1e-4_bs_16_epoch200/best.pth \
    OUTPUT_DIR results/fcntacotest'

