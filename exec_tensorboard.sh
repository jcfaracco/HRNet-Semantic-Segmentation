#! /bin/bash

docker run --name tensorboard -it --rm \
   -v $PWD:/tmp/hrnet \
    -p 0.0.0.0:6006:6006 \
    hrnet  \
    bash

