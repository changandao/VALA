#!/bin/bash
# Train 3DGS on ScanNet scenes and assign language features to the Gaussians.
export CUDA_VISIBLE_DEVICES=0

scan_names="scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00"
OUTPUT_DIR="scannet_langsplat"

for case in $scan_names
do
    echo "Processing $case"

    python train.py -s dataset/3dgs/scannet_langsplat/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000 --eval

    python gaussian_feature_extractor.py -m output/3dgs/$OUTPUT_DIR/$case -s dataset/3dgs/scannet_langsplat/$case \
        --iteration 30000 --eval --feature_level 0 --use_efficient --tau_mass 0.9 --tau_abs 0.01
done
