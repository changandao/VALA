python eval_scannet/eval_scannet_render.py --source_path dataset/3dgs/scannet_langsplat \
        --model_path output/3dgs/scannet_langsplat --ae-ckpt ckpt/ins \
        --label-map autolabel/label_map.csv \
        --vis render_langsplat/vis/occamlgs_stochastic  \
        --out render_langsplat/metrics/occamlgs_stochastic