#!/bin/sh
kernels_src_dir=$1
kernels_hex_dir=$2

cd $kernels_src_dir
for kernel_src in *.cl; do
  xxd -i $kernel_src>$kernels_hex_dir/$kernel_src.inc
done