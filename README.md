# WildHands: 3D Hand Pose Estimation in Everyday Egocentric Images

## Installation

First, install Python 3.10:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv
```

Then create and activate the virtual environment:
```bash
python3.10 -m venv .wildhands
source .wildhands/bin/activate
```

Fetch the submodules (ViTPose):
```bash
git submodule update --init --recursive
```

Then install the rest of the dependencies:
```bash
pip install -e .[all]
pip install -v -e third-party/ViTPose
pip install easydict
```

Install PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install PyTorch3D:
```bash
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
```

You also need to download the trained models:
```bash
bash fetch_models.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section. WildHands requires both `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`. You need to put them under the `downloads/wildhands` folder.

## Demo

Some example images are provided in the `downloads/example_data` folder. The code also requires the camera focal length to get 3D predictions. The default value is set to 1000 which works for the provided example images. WildHands requires the focal length as input to the network, and this model is trained on egocentric data only.

```bash
python demo.py
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
