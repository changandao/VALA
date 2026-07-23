#!/bin/bash
# Extract SAM masks and CLIP language features for the LERF_OVS dataset.
# Expects the SAM checkpoint at ckpts/sam_vit_h_4b8939.pth (override with --sam_checkpoint).
python run_sam.py --dataset_name lerf_ovs --rep 3dgs --get_semantic --use_langsplat
