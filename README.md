## Installation

First, install Python 3.10:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv
```

Then create and activate the virtual environment:
```bash
python3.10 -m venv .hands
source .hands/bin/activate
```

Then, you can install the rest of the dependencies. This is for CUDA 11.7, but you can adapt accordingly:
```bash
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
pip install easydict
```

Download the trained models:
```bash
bash fetch_models.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section. HaMeR requires only the right hand model `MANO_RIGHT.pkl` (put it under the `downloads/_DATA/data/mano` folder). WildHands requires both `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` (put them under the `downloads/wildhands` folder).

Set the required paths:
```bash
export CACHE_DIR_HAMER=downloads/_DATA
export HAMER_MANO_DIR=downloads/_DATA/data
export WILDHANDS_MANO_DIR=downloads/wildhands
export INTRX_PATH=downloads/wildhands/intrx.pkl
```

## Usage

Some example images are provided in the `downloads/example_data` folder. The code also requires the camera focal length to get 3D predictions. The default value is set to 1000 which works for the provided example images.

HaMeR assumes a focal length of 5000 in its predictions. It needs to be changed to the focal length of the camera used to capture the images to get accurate 3D predictions.
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --img_folder downloads/example_data --out_folder out --hamer_ckpt downloads/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
```

WildHands requires the focal length as input to the network. This model is trained on egocentric data only.
```
CUDA_VISIBLE_DEVICES=0 python demo.py --img_folder downloads/example_data --out_folder out --focal_length 1000 --wildhands_ckpt downloads/wildhands/wildhands.ckpt
```

## Acknowledgements
Check out these amazing repos as well which form the basis of this codebase:
- [HaMeR](https://github.com/geopavlakos/hamer)
- [ARCTIC](https://github.com/zc-alexfan/arctic)

## Citing
If you find this code useful, please consider citing:

```bibtex
@inproceedings{Prakash2024Hands,
    author = {Prakash, Aditya and Tu, Ruisen and Chang, Matthew and Gupta, Saurabh},
    title = {3D Hand Pose Estimation in Everyday Egocentric Images},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2024}
}

@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```