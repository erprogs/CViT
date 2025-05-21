import os
import cv2
import torch
import dlib
import face_recognition
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
from helpers.loader import *
from facenet_pytorch import MTCNN
from decord import VideoReader, cpu
from helpers.helpers_read_video_1 import VideoReader as VR
from helpers.helpers_face_extract_1 import FaceExtractor
from helpers.blazeface import BlazeFace

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.empty_cache()

mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)

facedet = BlazeFace().to(device)
facedet.load_weights("helpers/blazeface.pth")
facedet.load_anchors("helpers/anchors.npy")
_ = facedet.train(False)

def load_cvit(cvit_weight, net, fp16):
    
    from model.cvit import CViT as cvit # load cvit2
    if net == 'cvit':
        from model.cvit_old import CViT as cvit

    model = cvit(image_size=224, patch_size=7, num_classes=2, channels=512,
             dim=1024, depth=6, heads=8, mlp_dim=2048)

    model.to(device)
    checkpoint = torch.load(os.path.join("weight", cvit_weight), map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    _ = model.eval()

    if fp16:
        model.half()

    return model

def face_mtcnn_(frame):
    boxes, con = mtcnn.detect(frame)
    return boxes is not None and con[0]>0.95


def face_mtcnn(frames):
    padding = 0
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0

    for _, frame in tqdm(enumerate(frames), total=len(frames)):

        try:
            boxes, conf = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    if count < 5:
                        # Extract coordinates
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(frame.shape[1], x2 + padding)  # frame.shape[1] is the width of the frame
                        y2 = min(frame.shape[0], y2 + padding)  # frame.shape[0] is the height of the frame
                        
                        # Crop the face from the frame
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            # Resize the cropped face and convert it back to BGR for consistency with OpenCV
                            resized_face = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
                            resized_face_bgr = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                            temp_face[count] = resized_face_bgr
                            count += 1
        except:
            print('error encountered when extracting video frames')

    return ([], 0) if count == 0 else (temp_face[:count], count)

def face_blaze(video_path):

    frames_per_video = 15 
    video_reader = VR()
    video_read_fn = lambda x: video_reader.read_random_frames(x, num_frames=frames_per_video)
    try:
        face_extractor = FaceExtractor(video_read_fn, facedet)
        
        faces = face_extractor.process_video(video_path)
        # Only look at one face per frame.
        #face_extractor.keep_only_best_face(faces)
        
        count_blaze=0
        temp_blaze = np.zeros((15, 224, 224, 3), dtype=np.uint8)
        for frame_data in faces:
            for face in frame_data["faces"]:
                if count_blaze<14:
                    resized_facefrm = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_facefrm = cv2.cvtColor(resized_facefrm, cv2.COLOR_RGB2BGR)
                    crop_amount = 30  # Adjust this value as needed

                    # Get the original dimensions of the image
                    height, width, _ = resized_facefrm.shape

                    # Calculate the new dimensions after cropping
                    new_height = height - 2 * crop_amount
                    new_width = width - 2 * crop_amount

                    # Perform cropping
                    cropped_image = resized_facefrm[crop_amount:new_height+crop_amount, crop_amount:new_width+crop_amount]
                    resized_facefrm_cropped = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_AREA)
                    temp_blaze[count_blaze]=resized_facefrm_cropped

                    count_blaze+=1
    
    except:
        print('error encountered when extracting video frames')

    return ([], 0) if count_blaze == 0 else (temp_blaze[:count_blaze], count_blaze)

def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"
    padding = 10
    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            face_locations = face_recognition.face_locations(
                frame, number_of_times_to_upsample=0, model=mod
            )

            #print('len',len(face_locations))

            for face_location in face_locations:
                if count < len(frames):
                    top, right, bottom, left = face_location
                    top = max(0, top - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    left = max(0, left - padding)
                    right = min(frame.shape[1], right + padding)
                        
                    face_image = frame[top:bottom, left:right]
                    face_image = cv2.resize(
                        face_image, (224, 224), interpolation=cv2.INTER_AREA
                    )

                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                    #if face_mtcnn_(face_image):
                    #    print('FILTEINRG IMAGE WITH MTCNN')
                    #    temp_face[count] = face_image
                    #    count += 1

                    temp_face[count] = face_image
                    count += 1
                else:
                    break
        except:
            print('error encountered when extracting video frames')

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    df_tensor = torch.tensor(frame, device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)

    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):
    # Finds the index and value of the maximum prediction value.
    mean_val = torch.mean(y_pred, dim=0)

    if mean_val.numel() == 1:
        mean_val = y_pred

    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // frames_nums)  # Calculate the step size between frames
    return vr.get_batch(
        list(range(0, len(vr), step_size))[:frames_nums]
    ).asnumpy()  # seek frames with step_size


def df_face(vid, num_frames):
    img = extract_frames(vid, num_frames)
    face, count = face_rec(img)
    #face, count = face_mtcnn(img)
    #face, count = face_blaze(vid)
    return preprocess_frame(face) if count > 0 else []


def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result
