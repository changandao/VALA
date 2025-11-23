export CUDA_VISIBLE_DEVICES=0
# case="teatime"
# case="figurines"
# case="waldo_kitchen ramen"
# case="waldo_kitchen"
case="ramen"
# case="figurines"teatime figurines 
cases="teatime"
cases="figurines ramen waldo_kitchen"
# cases="ramen"

OUTPUT_DIR="lerf_ovs_occamlgs_ablation"

# for case in $cases
# do
# # python train.py -s dataset/3dgs/lerf_ovs/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000
# # python render.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000
# python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
# --iteration 30000 --feature_level 0 --ablation_type none --eval
# # --omniseg3d --use_efficient
# done

for case in $cases
do
    feature_levels="1 2 3"
    for i in $feature_levels
    do
        echo "Processing feature level $i"
        # python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
        # --iteration 30000  --feature_level $i --ablation_type none --eval

        python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval \
        --feature_level $i --skip_train --ablation_type gate_weights
    done
done