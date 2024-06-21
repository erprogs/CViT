# CViT
### Deepfake Video Detection Using Convolutional Vision Transformer

# Update, April 1, 2024
------------------------------------------------------------------------
# CViT2
### Improved Deepfake Video Detection Using Convolutional Vision Transformer

Implementation code for our paper. 
[link to paper](https://arxiv.org/abs/2102.11126) | [link to MS Thesis](http://etd.aau.edu.et/handle/123456789/24209) | [link to MS Thesis defense PPT file](https://github.com/erprogs/CViT/blob/main/CViT.pptx)      | [link to CViT2](comingsoon)

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

```python cvit_prediction.py --p <video path> --f <number_of_frames>  --w <weights_path> --n <network_type> --fp16 <half_precision>```

### To predict on some deepfake datasets:

```python cvit_prediction.py --p <video path> --f <number_of_frames> --d <dataset_type> --w <weights_path> --n <network_type> --fp16 <half_precision>```

E.g usage:

```python cvit_prediction.py --p sample__prediction_data --f 15 --n cvit2 --fp16 y ```


#### Arguments

    Predicts whether a video is Deepfake or not.<br />
    Prediction value <0.5 - REAL <br />
    Prediction value >=5  - FAKE

    --p (str): Path to the video or image file for prediction.
        Example: --p /path/to/video.mp4

    --f (int): Number of frames to process for prediction.
        Example: --f 30

    --d (str): Dataset type. Options are dfdc, faceforensics, timit, or celeb.
        Example: --d dfdc

    --w (str): Path to the model weights for CViT or CViT2.
        Example: --w /path/to/weights.pth

    --n (str): Network type. Options are cvit or cvit2.
        Example: --n cvit

    --fp16 (str): Enable half-precision support. Accepts a boolean value (true or false).
        Example: --fp16 true


### Train CViT
To train the model on your own you can use the following parameters:<br />

``` python train_cvit.py -e <epochs> --d <data_path> --b <batch_size> --l <learning_rate> --w <weight_decay> --t <test_option>```

### Options

    -e, --epoch (int): Number of training epochs, defualt=1.
    -d, --dir (str): Path to the training data.
    -b, --batch (int): Batch size, defualt=32.
    -l, --rate (float): Learning rate, default=0.001.
    -w, --wdecay (float): Weight decay, default= 0.0000001 .
    -t, --test (str): Test on test set (e.g., y).


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
    author={Deressa Wodajo and Peter Lambert and Glenn Van Wallendael and Solomon Atnafu and Hannes Mareen},
    booktitle={Proceedings of the IEEE International Conference on Games, Entertainment & Media (GEM)},
    year={2024},
    month={June},
    address={Turin (Torino), Italy}
}
```

