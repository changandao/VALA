# Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://vala3d.github.io/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2509.05515)
[![arXiv](https://img.shields.io/badge/arXiv-2509.05515-b31b1b.svg)](https://arxiv.org/abs/2509.05515)

> Official implementation of **Visibility-Aware Language Aggregation (VALA)**.

## Inatallation

First create the environment and install the required dependencies

```
conda create -n vala python=3.10 

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

pip install -e ./submodules/gsplat
pip install -e ./submodules/simple-knn

```

## Running the code 
### Step1: getting the dataaset and extracting the language features.
Please put the downloaded dataset as following hiearchy:

First, create a `dataset/` folder.

### LERF_OVS

Download the [dataset](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view) and upzip to `dataset/lerf_ovs`. The data structure will be organized as follow:
```
dataset/
├── lerf_ovs
│   ├── label/
│   │   ├── figurines/
│   │   │   ├── frame_000041.jpg
│   │   │   ├── frame_000041.json
│   │   │   ├── ...
│   │   ├── ramen/
│   │   │   ├── frame_000006.jpg
│   │   │   ├── frame_000006.json
│   │   │   ├── ...
│   │   └── ...
│   ├── figurines/
│   │   ├── images
│   │   │   ├── frame_00001.jpg
│   │   │   ├── frame_00002.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── ramen/
│   │   ├── images
│   │   │   ├── frame_00001.jpg
│   │   │   ├── frame_00002.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

Nest, you will need to extract SAM + CLIP features.

Change the ```--root_dir``` in scripts/run_sam.sh to ```path/to/project```

then run
```
sh scripts/run_sam.sh
```

Note that we current support lerf_ovs, scannet, and waymo datasets, for mipnerf360 and replica we cannot guarantee its efffeciveness.

### Step2: reconstruct the scene with 3d Gaussian Splatting and assign the language feuatres to the 3D Gaussians

```
sh scripts/run_lerf_3d.sh
```

After running this, you will get the trained 3D Gasussian Splatting and the 3 levels language features assgined to each gaussian.

### Step3: running evaluation

```
sh scripts/eval_lerf_ovs_3d.sh
```

## Related Projects

We also refer readers to our related project:

- [SuperGSeg](https://github.com/supergseg/supergseg) — A complementary work on open-vocabulary 3D segmentation using Gaussian-based scene representations.

## Acknowledgements

This project builds upon several excellent open-source works. We would like to express our sincere gratitude to the authors and contributors of:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [OccamLGS](https://github.com/insait-institute/OccamLGS)

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



