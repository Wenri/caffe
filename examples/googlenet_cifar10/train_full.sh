#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe.bin train \
    --solver=examples/googlenet_cifar10/solver.prototxt
