#!/usr/bin/env sh

TOOLS=./build/tools

#$TOOLS/caffe.bin train \
#    --solver=examples/cifar10/cifar10_toep_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe.bin train \
    --solver=examples/cifar10/cifar10_toep_solver_lr1.prototxt \
    --snapshot=examples/cifar10/cifar10_toep_iter_60000.solverstate.h5

# reduce learning rate by factor of 10
$TOOLS/caffe.bin train \
    --solver=examples/cifar10/cifar10_toep_solver_lr2.prototxt \
    --snapshot=examples/cifar10/cifar10_toep_iter_65000.solverstate.h5
