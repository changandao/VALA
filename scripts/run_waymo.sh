export CUDA_VISIBLE_DEVICES=0
# case="teatime"
# case="figurines"
# case="waldo_kitchen ramen"
# case="waldo_kitchen"
case="waymo_000000"
# case="figurines"teatime figurines 
# cases="teatime"
cases="teatime figurines ramen waldo_kitchen"
cases="1534950_1cams"
OUTPUT_DIR="waymo"

for case in $cases
do
python train.py -s dataset/3dgs/waymo/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000
# python render.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000

python gaussian_feature_extractor.py -m output/3dgs/$OUTPUT_DIR/$case \
--iteration 30000 --eval --feature_level 1 --use_efficient

# python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval --feature_level 1


done