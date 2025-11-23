export CUDA_VISIBLE_DEVICES=1
# case="teatime"
# case="figurines"
# case="waldo_kitchen ramen"
# case="waldo_kitchen"
case="ramen"
# case="figurines"teatime figurines 
cases="teatime"
# cases="teatime figurines ramen waldo_kitchen"
OUTPUT_DIR="lerf_ovs_occamlgs_ablation"
ablation_type="rpx30_vala_wogate"
for case in $cases
do
for feature_level in 3
do
# # python train.py -s dataset/3dgs/lerf_ovs/$case -m output/3dgs/$OUTPUT_DIR/$case --iterations 30000
# # python render.py -m output/3dgs/$OUTPUT_DIR/$case --iteration 30000
# python gaussian_feature_extractor_ablation.py -m output/3dgs/$OUTPUT_DIR/$case \
# --iteration 30000 --feature_level 0 --use_efficient --omniseg3d --ablation_type omniseg3d --eval
python caculate_disp.py -m output/3dgs/$OUTPUT_DIR/$case \
--iteration 30000 --feature_level $feature_level --use_efficient \
--ablation_type $ablation_type --calculate_dispersion --stage 2  
done
# python caculate_disp.py --analyze_dispersion output/3dgs/$OUTPUT_DIR/$case/$ablation_type/dispersion_30000_omniseg3d_omniseg3d.pth
done

