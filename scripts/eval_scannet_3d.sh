# for target_nums in 10 15 19
# do
#     python eval_scannet_3d.py --iteration 30000 --ablation_type quantile85_gate --target_nums $target_nums
# done

for target_nums in 19
do
    python eval_scannet_3d.py --iteration 30000 --ablation_type adaptive_gate --target_nums $target_nums
done