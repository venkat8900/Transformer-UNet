# Bio-Medical Image Segmentation. 

## Dataset
1. MICCAI 2017: Robotic Tool Segmentation https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/ 

The training dataset consists of 8 x 225-frame sequences of high resolution stereo camera images acquired from a da Vinci Xi surgical system during several different porcine procedures. Every video sequence consists of two stereo channels taken from left and right cameras and has a 1920 x 1080 pixel resolution in RGB format. The articulated parts of the robotic surgical instruments, such as a rigid shaft, an articulated wrist and claspers have been hand labelled in each frame. Furthermore, there are instrument type labels that categorize instruments in following categories: left/right prograsp forceps, monopolar curved scissors, large needle driver, and a miscellaneous category for any other surgical instruments.

* Images are cropped from (320, 28) to extract 1280 x 1024 dimension camera images.

* Three Sub Challenges: Binary Instrument Segmentation, Instrument Part Segmentation, Instrument Type Segmentation.

* Based on the mappings, we generate ground truth masks to train the model. 

* Dataset: https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org

## Dependencies
* Python 3.6
* PyTorch 0.4.0
* TorchVision 0.2.1
* numpy 1.14.0
* opencv-python 3.3.0.10
* tqdm 4.19.4

To install all these dependencies, run
```
pip install -r requirements.txt
```

# Usage

# Squeeze and Excitations

```
cd "Squeeze_Excitation"
```

Directory structure is as follows
```
    ├── Squeeze_Excitation
    ├── data
    │   ├── RTS_binary
    │   ├── RTS_instrument
    │   ├── RTS_parts
    │   ├── RTS
    │   │   ├── test
    |   │   │   ├── instrument_dataset_1
    |   │   │   │   ├── left_frames
    |   │   │   │   └── right_frames
    |   |   |   ....................... 
    │   |   └── train
    |   │   |   ├── instrument_dataset_1
    |   │   |   │   ├── ground_truth
    |   │   |   │   │   ├── Left_Prograsp_Forceps_labels
    |   │   |   │   │   ├── Maryland_Bipolar_Forceps_labels
    |   │   |   │   │   ├── Other_labels
    |   │   |   │   │   └── Right_Prograsp_Forceps_labels
    |   │   |   │   ├── left_frames
    |   │   |   │   └── right_frames
    |   │   |   .......................
```

# 1. Preprocessing

As a preprocessing step we cropped black unindormative border from all frames with a file ``prepare_data.py`` that creates folder ``data/cropped_train.py`` with masks and images of the smaller size that are used for training. Then, to split the dataset for 4-fold cross-validation one can use the file: ``prepare_train_val``.


# 2. Training

The main file that is used to train all models -  ``train.py``.

Running ``python train.py --help`` will return set of all possible input parameters.

To train all models we used the folloing bash script :

```
    #!/bin/bash

    for i in 0 1 2 3
    do
       python train.py --device-ids 0,1,2,3 --batch-size 16 --fold $i --workers 12 --lr 0.0001 --n-epochs 10 --type binary --jaccard-weight 1 --model UNetCSE
    done
```

# 3. Mask generation

The main file to generate masks is ``generate_masks.py``.

Running ``python generate_masks.py --help`` will return set of all possible input parameters.

Example:
```
    python generate_masks.py --output_path predictions/unet/binary --model_type UNet --problem_type binary --model_path runs/debug/unet/binary --fold -1 --batch-size 4
```

# 4. Evaluation

The evaluation is different for a binary and multi-class segmentation: 

[a] In the case of binary segmentation it calculates jaccard (dice) per image / per video and then the predictions are avaraged. 

[b] In the case of multi-class segmentation it calculates jaccard (dice) for every class independently then avaraged them for each image and then for every video
```
    python evaluate.py --target_path predictions/unet --problem_type binary --train_path data/cropped_train
```

# TransUNet
### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* Base code taken from: https://github.com/Beckschen/TransUNet


