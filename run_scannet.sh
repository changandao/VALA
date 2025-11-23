export CUDA_VISIBLE_DEVICES=1
scan_names="scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00"
scan_names="scene0000_00"
# scan_names="scene0062_00 scene0070_00 scene0097_00 scene0140_00  scene0347_00 scene0400_00 scene0590_00 "
scan_names="scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00  scene0347_00 scene0400_00 scene0590_00"

scan_names="scene0050_02"
OUTPUT_DIR="scannet_langsplat_large_dgrad_0004"
OUTPUT_DIR="scannet_langsplat"

for case in $scan_names
do
# python train.py -s dataset/3dgs/scannet_langsplat/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000 --eval 
# python render.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --skip_train --eval

python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case -s dataset/3dgs/scannet_langsplat/$case \
--iteration 30000 --eval --feature_level 0 --omniseg3d --ablation_type adaptive_gate --use_efficient

# python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval --feature_level 0 --skip_train
done

# feature_levels="1 2 3"
# for i in $feature_levels
# do
#     echo "Processing feature level $i"
#     python gaussian_feature_extractor.py -m output/3dgs/$OUTPUT_DIR/$case \
#     --iteration 30000 --eval --feature_level $i #--use_efficient
#     # python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval --feature_level $i
# done