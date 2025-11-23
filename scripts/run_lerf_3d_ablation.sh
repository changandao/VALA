export CUDA_VISIBLE_DEVICES=1
# case="teatime"
# case="figurines"
# case="waldo_kitchen ramen"
# case="waldo_kitchen"
case="ramen"
# case="figurines"teatime figurines 
# cases="teatime"
cases="teatime figurines ramen waldo_kitchen"
OUTPUT_DIR="lerf_ovs_occamlgs_ablation"

# for case in $cases
# do
# # python train.py -s dataset/3dgs/lerf_ovs/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000
# # python render.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000
# python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
# --iteration 30000 --feature_level 0 --use_efficient --omniseg3d --ablation_type omniseg3d --eval

# done

for case in $cases
do
    feature_levels="1 2 3"
    for i in $feature_levels
    do
        echo "Processing feature level $i"
        python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
        --iteration 30000  --feature_level $i --ablation_type ablation_stageA_plus_B --eval --use_efficient
        # --iteration 30000  --feature_level $i --ablation_type ablation_stageA_mass_coverage --eval --use_efficient
        
    # python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval --feature_level $i
    done
done


# ## occamlgs
# for case in $cases
# do
#     feature_levels="1 2 3"
#     for i in $feature_levels
#     do
#         echo "Processing feature level $i"
#         python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
#         --iteration 30000  --feature_level $i --ablation_type occamlgs --eval
#     # python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval --feature_level $i
#     done
# done

