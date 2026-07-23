# Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://vala3d.github.io/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2509.05515)
[![arXiv](https://img.shields.io/badge/arXiv-2509.05515-b31b1b.svg)](https://arxiv.org/abs/2509.05515)

> Official implementation of **Visibility-Aware Language Aggregation (VALA)**.

## Installation

First create the environment and install the required dependencies:

```bash
conda create -n vala python=3.10
conda activate vala

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Then clone the submodules and install them:

```bash
git submodule update --init --recursive

pip install -e ./submodules/gsplat
pip install -e ./submodules/simple_knn
pip install -e ./submodules/segment-anything-langsplat
```

Finally, download the SAM checkpoint used for mask extraction and place it under `ckpts/`:

```bash
mkdir -p ckpts
wget -P ckpts https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Running the code

### Step 1: Get the dataset and extract the language features

#### LERF_OVS

Download the [dataset](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view) and unzip it to `dataset/3dgs/lerf_ovs`. The data structure should be organized as follows:

```
dataset/
├── 3dgs/
│   └── lerf_ovs/
│       ├── label/
│       │   ├── figurines/
│       │   │   ├── frame_000041.jpg
│       │   │   ├── frame_000041.json
│       │   │   └── ...
│       │   ├── ramen/
│       │   │   ├── frame_000006.jpg
│       │   │   ├── frame_000006.json
│       │   │   └── ...
│       │   └── ...
│       ├── figurines/
│       │   ├── images/
│       │   │   ├── frame_00001.jpg
│       │   │   ├── frame_00002.jpg
│       │   │   └── ...
│       │   └── sparse/
│       │       └── 0/
│       ├── ramen/
│       │   ├── images/
│       │   │   ├── frame_00001.jpg
│       │   │   ├── frame_00002.jpg
│       │   │   └── ...
│       │   └── sparse/
│       │       └── 0/
│       └── ...
```

Next, extract the SAM + CLIP language features:

```bash
sh scripts/run_sam.sh
```

The script assumes the SAM checkpoint is at `ckpts/sam_vit_h_4b8939.pth`; pass `--sam_checkpoint` to `run_sam.py` to override.

We currently support the LERF_OVS and ScanNet datasets (see `run_scannet.sh` for the ScanNet pipeline). Waymo is supported experimentally; for MipNeRF360 and Replica we cannot guarantee effectiveness.

### Step 2: Reconstruct the scene with 3D Gaussian Splatting and assign the language features to the 3D Gaussians

```bash
sh scripts/run_lerf_3d.sh
```

After running this, you will get the trained 3D Gaussian Splatting reconstruction with three levels of language features assigned to each Gaussian.

### Step 3: Run the evaluation

```bash
sh scripts/eval_lerf_ovs_3d.sh
```

## Related Projects

We also refer readers to our related project:

- [SuperGSeg](https://github.com/supergseg/supergseg) — A complementary work on open-vocabulary 3D segmentation using Gaussian-based scene representations.

## License

This project builds upon [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and is distributed under the same Gaussian-Splatting license (free for non-commercial research and evaluation use); see [LICENSE.md](LICENSE.md).

## Acknowledgements

This project builds upon several excellent open-source works. We would like to express our sincere gratitude to the authors and contributors of:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [OccamLGS](https://github.com/insait-institute/OccamLGS)
- [LangSplat](https://github.com/minghanqin/LangSplat)

Their outstanding contributions to efficient Gaussian splatting implementations have significantly supported this work.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{wang2025vala,
  title={Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting},
  author={Sen Wang and Kunyi Li and Siyun Liang and Elena Alegret and Jing Ma and Nassir Navab and Stefano Gasperini},
  year={2025},
  eprint={2509.05515},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.05515}
}
```
