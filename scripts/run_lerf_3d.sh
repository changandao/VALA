export CUDA_VISIBLE_DEVICES=0
cases="teatime figurines ramen waldo_kitchen"
OUTPUT_DIR="lerf_ovs"

for case in $cases
do
    python train.py -s dataset/3dgs/lerf_ovs/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000

    feature_levels="1 2 3"
    for i in $feature_levels
    do
        echo "Processing feature level $i"
        python gaussian_feature_extractor.py -m output/3dgs/$OUTPUT_DIR/$case \
        --iteration 30000  --feature_level $i --use_efficient --eval
    done

done


