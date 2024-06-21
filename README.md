# CViT
### Deepfake Video Detection Using Convolutional Vision Transformer

Implementation code for our paper. 
[link to paper](https://arxiv.org/abs/2102.11126) | [link to MS Thesis](http://etd.aau.edu.et/handle/123456789/24209) | [link to MS Thesis defense PPT file](https://github.com/erprogs/CViT/blob/main/CViT.pptx)      | [link to CViT2](comingsoon)


## Update, April 1, 2024

# CViT2
### Improved Deepfake Video Detection Using Convolutional Vision Transformer


### Requirements:
* Pytorch >=1.4

### DL library used for face extraction
   * helpers_read_video_1.py
   * helpers_face_extract_1.py
   * blazeface.py
   * blazeface.pth
   * face_recognition
   * facenet-pytorch
   * dlib

### Preprocessing

extractfaces.py
    Face extraction from video.
    The code works for DFDC dataset. You can test it using the sample data provided.

### Weights
cvit_deepfake_detection_ep_50.pth - Model weight for CViT. <br />
cvit2_deepfake_detection_ep_50.pth - Model weight for CViT2. <br />

### Predict CViT 
Download the pretrained model from [Huggingface](https://huggingface.co/datasets/Deressa/cvit) and save it in the `weight` folder.

##### CViT2 - trained on 5 datasets including DFDC

```bash
wget https://huggingface.co/datasets/Deressa/cvit/blob/main/cvit2_deepfake_detection_ep_50.pth
```
or 

##### CViT - trained on DFDC

```bash
wget https://huggingface.co/datasets/Deressa/cvit/blob/main/cvit_deepfake_detection_ep_50.pth
```

```python cvit_prediction.py --p <video path> --f <number_of_frames>  --w <weights_path> --n <network_type> --fp16 <half_precision>```

### To predict on some deepfake datasets:

```python cvit_prediction.py --p <video path> --f <number_of_frames> --d <dataset_type> --w <weights_path> --n <network_type> --fp16 <half_precision>```

E.g usage:

```python cvit_prediction.py --p sample__prediction_data --f 15 --n cvit2 --fp16 y ```


#### Arguments

Predicts whether a video is Deepfake or not.<br />
Prediction value <0.5 - REAL <br />
Prediction value >=5  - FAKE

&nbsp;&nbsp;&nbsp;&nbsp; --p (str): Path to the video or image file for prediction.

&nbsp;&nbsp;&nbsp;&nbsp;    Example: --p /path/to/video.mp4

&nbsp;&nbsp;&nbsp;&nbsp; --f (int): Number of frames to process for prediction.

&nbsp;&nbsp;&nbsp;&nbsp;    Example: --f 30

&nbsp;&nbsp;&nbsp;&nbsp; --d (str): Dataset type. Options are dfdc, faceforensics, timit, or celeb.

&nbsp;&nbsp;&nbsp;&nbsp;    Example: --d dfdc

&nbsp;&nbsp;&nbsp;&nbsp; --w (str): Path to the model weights for CViT or CViT2.

&nbsp;&nbsp;&nbsp;&nbsp;    Example: --w cvit2_deepfake_detection_ep_50

&nbsp;&nbsp;&nbsp;&nbsp; --n (str): Network type. Options are cvit or cvit2.

&nbsp;&nbsp;&nbsp;&nbsp;    Example: --n cvit

&nbsp;&nbsp;&nbsp;&nbsp; --fp16 (str): Enable half-precision support. Accepts a boolean value (true or false).

&nbsp;&nbsp;&nbsp;&nbsp;    Example: --fp16 true


### Train CViT
To train the model on your own you can use the following parameters:<br />

``` python train_cvit.py -e <epochs> --d <data_path> --b <batch_size> --l <learning_rate> --w <weight_decay> --t <test_option>```

### Options

&nbsp;&nbsp;&nbsp;&nbsp; -e, --epoch (int): Number of training epochs, defualt=1.

&nbsp;&nbsp;&nbsp;&nbsp; -d, --dir (str): Path to the training data.

&nbsp;&nbsp;&nbsp;&nbsp; -b, --batch (int): Batch size, defualt=32.

&nbsp;&nbsp;&nbsp;&nbsp; -l, --rate (float): Learning rate, default=0.001.

&nbsp;&nbsp;&nbsp;&nbsp; -w, --wdecay (float): Weight decay, default= 0.0000001.

&nbsp;&nbsp;&nbsp;&nbsp; -t, --test (str): Test on test set (e.g., y).


### Authors
**Deressa Wodajo** <br />
**Solomon Atnafu** <br />
**Peter Lambert** <br />
**Glenn Van Wallendael** <br />
**Hannes Mareen** <br />

## Bibtex
#### CViT
```bash
@misc{wodajo2021deepfake,
      title={Deepfake Video Detection Using Convolutional Vision Transformer}, 
      author={Deressa Wodajo and Solomon Atnafu},
      year={2021},
      eprint={2102.11126},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
#### CViT2
```bash
@inproceedings{wodajo2024deepfake,
    title={Improved Deepfake Video Detection Using Convolutional Vision Transformer},
    author={Deressa Wodajo, Peter Lambert, Glenn Van Wallendael, Solomon Atnafu and Hannes Mareen},
    booktitle={Proceedings of the IEEE International Conference on Games, Entertainment & Media (GEM)},
    year={2024},
    month={June},
    address={Turin (Torino), Italy}
}
```

