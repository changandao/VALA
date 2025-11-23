export CUDA_VISIBLE_DEVICES=1
cases="teatime figurines ramen waldo_kitchen"
cases="teatime"
OUTPUT_DIR="lerf_ovs"

# for case in $cases
# do
# python train.py -s dataset/3dgs/lerf_ovs/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000
# # python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
# # --iteration 30000 --feature_level 0 --use_efficient --omniseg3d --ablation_type omniseg3d --eval

# done

for case in $cases
do
    feature_levels="1 2 3"
    for i in $feature_levels
    do
        echo "Processing feature level $i"
        python gaussian_feature_extractors.py -m output/3dgs/$OUTPUT_DIR/$case \
        --iteration 30000  --feature_level $i --use_efficient --eval
    # python feature_map_renderer.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000 --eval --feature_level $i
    done
done


