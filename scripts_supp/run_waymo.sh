export CUDA_VISIBLE_DEVICES=0
# case="teatime"
# case="figurines"
# case="waldo_kitchen ramen"
# case="waldo_kitchen"
# case="figurines"teatime figurines 
# cases="teatime"
# cases="teatime figurines ramen waldo_kitchen"
cases="1534950_1cams"
OUTPUT_DIR="waymo"


for case in $cases
do
    feature_levels="1 2 3"
    for i in $feature_levels
    do
        echo "Processing feature level $i"
        python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
        --iteration 30000  --feature_level $i --ablation_type cos_median --use_efficient 
        # --iteration 30000  --feature_level $i --ablation_type ablation_stageA_mass_coverage --eval --use_efficient
    # python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval --feature_level $i
    done
done
