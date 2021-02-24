# CViT
### Deepfake Video Detection Using Convolutional Vision Transformer

Implementation code for our paper. 
[link to paper](https://arxiv.org/abs/2102.11126)

### Requirements:
* Pytorch 1.4

### DL library used for face extraction
   * helpers_read_video_1.py
   * helpers_face_extract_1.py
   * blazeface.py
   * blazeface.pth

### Weights
1. cvit_nov16_Auged_90p_50ep.pth <br/>
   CViT model weight, November 16, 2020, Augmentation=90%, Epoch=50.

2. Deepfake_Oct6_Auged_90p_50ep.pth <br/>
   Deep CNN Deepfake detection weight - ~87% accuracy on DFDC dataset (not documented in the thesis)

### Preprocessing
extractfaces.py
   The code works for DFDC dataset. You can test it using the sample data provided. 

#### Predict CViT 

python CViT_prediction.py
   Predicts whether a video is Deepfake or not.
   <0.5 - REAL
   >=5  - FAKE


### Train CViT
e: epoch
s: session - GPU or TPU
w: weight decay  default= 0.0000001
l: learning rate defualt=0.001
d: path file
b: batch size, defualt=32

python cvit_train.py -e 10 -s 'g' -l 0.001 -w 0.0000001 -d sample_train_data/
