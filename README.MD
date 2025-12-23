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



