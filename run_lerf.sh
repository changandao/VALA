DATASET_NAME="teatime"
OUTPUT_DIR="/home/joanna_cheng/workspace/occamlgs/output/lerf"

cd ~/workspace/occamlgs

python train.py -s /scratch/joanna_cheng/lerf_ovs/$DATASET_NAME -m $OUTPUT_DIR/$DATASET_NAME --iterations 30000
python render.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000

python gaussian_feature_extractor.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000 --eval --feature_level 1
python feature_map_renderer.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000 --eval --feature_level 1