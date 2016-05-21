#!/usr/bin/env sh

TOOLS=./build/tools

#$TOOLS/caffe.bin train \
#    --solver=examples/cifar10/cifar10_acdc_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe.bin train \
    --solver=examples/cifar10/cifar10_acdc_solver_lr1.prototxt \
    --snapshot=examples/cifar10/cifar10_acdc_iter_60000.solverstate.h5

# reduce learning rate by factor of 10
$TOOLS/caffe.bin train \
    --solver=examples/cifar10/cifar10_acdc_solver_lr2.prototxt \
    --snapshot=examples/cifar10/cifar10_acdc_iter_65000.solverstate.h5
