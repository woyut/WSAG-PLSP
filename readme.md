# Weakly-Supervised Affordance Grounding Guided by Part-Level Semantic Priors

This is the PyTorch implementation of "Weakly-Supervised Affordance Grounding Guided by Part-Level Semantic Priors" (ICLR 2025).

## Data preparation

Please follow [LOCATE](https://github.com/Reagan1311/LOCATE) to prepare the AGD20K datasets. 

Please download the additional data (the initial and refined pseudo labels, the pre-extracted text features, the detected objects in the exocentric images, the recorded similarity for selecting exocentric images) at [here](https://drive.google.com/file/d/1m0A1ke7n2aplJXFqLJtV2vcDtrpKlCLc/view?usp=sharing), and merge the folder with the original AGD20K dataset.

Please download the pre-trained ViT-B/16 weights from [CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt), which is used to initialize the visual encoder during training.

## Prerequisites

The code is based on torch=1.10.0+cu111, torchvision=0.11.0+cu111, and numpy, opencv-python, PyYAML, tqdm, Pillow.


## Training

To train the model, run:

```
python train.py --config configs/seen.yaml --seed 10000
```

or

```
python train.py --config configs/unseen.yaml --seed 10000
```

for the seen/unseen split.

Please modify the paths in the .yaml files to the locations of the datasets and pre-trained ViT weights.

Our trained model can be found [here](https://drive.google.com/drive/folders/1JaX-4w9mH0IrxtKCowoBwCbD6loMyzKe?usp=sharing).

## Testing

To evaluate the trained model, run:

```
python eval.py --config configs/seen_test.yaml
```

or

```
python eval.py --config configs/unseen_test.yaml
```

Please modify the paths in the .yaml files to the locations of the datasets and the trained model weights.

## Pesudo Label Generation

To generate the pseudo labels for the egocentric images by yourself, please clone [grounded-segment-any-parts](https://github.com/Saiyan-World/grounded-segment-any-parts) and copy `codes/PL_generation/get_PL.py` to it. After preparing the environment following the instructions of grounded-segment-any-parts, you can run
```
python get_PL.py --split_name Seen/trainset --AGD_dir /path/to/AGD20K --output_name PL
```
and
```
python get_PL.py --split_name Unseen/trainset --AGD_dir /path/to/AGD20K --output_name PL
```
to generate the pseudo labels for the seen/unseen split. The generated masks will be saved under `AGD20K/Seen/trainset/PL` and `AGD20K/Unseen/trainset/PL`.

Similarly, you can copy `codes/PL_generation/get_obj_box.py` to grounded-segment-any-parts and generate the object bounding boxes for egocentric and exocentric images by
```
python get_obj_box.py --split_name Seen/trainset --view_name egocentric --AGD_dir /path/to/AGD20K --save_file_name ego_seen_box.pth
python get_obj_box.py --split_name Unseen/trainset --view_name egocentric --AGD_dir /path/to/AGD20K --save_file_name ego_unseen_box.pth
python get_obj_box.py --split_name Seen/trainset --view_name exocentric --AGD_dir /path/to/AGD20K --save_file_name exo_seen_box.pth
python get_obj_box.py --split_name Unseen/trainset --view_name exocentric --AGD_dir /path/to/AGD20K --save_file_name exo_unseen_box.pth
```

## Pseudo Label Refinement

coming soon


## Acknowledgement

We sincerely thank the codebase of [CLIP](https://github.com/openai/CLIP/), [SAM](https://github.com/facebookresearch/segment-anything/tree/main), [LOCATE](https://github.com/Reagan1311/LOCATE), and [grounded-segment-any-parts](https://github.com/Saiyan-World/grounded-segment-any-parts).
