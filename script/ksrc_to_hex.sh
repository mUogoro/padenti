#!/bin/sh

kernels_src_dir=$1
kernels_hex_dir=$2

cd $kernels_src_dir

for kernel_src in *.cl; do
  kernel_name=$(echo "$kernel_src"|cut -d . -f 1)
  touch $kernels_hex_dir/$kernel_src.inc
  echo "#ifndef ${kernel_name}_HPP"   > $kernels_hex_dir/$kernel_src.inc
  echo "#define ${kernel_name}_HPP"   >> $kernels_hex_dir/$kernel_src.inc
  xxd -i $kernel_src                      >> $kernels_hex_dir/$kernel_src.inc
  echo "#endif // ${kernel_name}_HPP" >> $kernels_hex_dir/$kernel_src.inc
done
