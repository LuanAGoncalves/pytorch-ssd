import buffer
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.vgg_ssd_new import create_vgg_ssd_new
from vision.ssd.predictor import PredictorM1, PredictorM2
from vision.ssd.config import vgg_ssd_config as config
import socket
import sys
import json
import torch
import os
import time
import argparse
import cv2
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-path", required=False,
        help="path to the trained model", type=str, default="./models/ufpark-model.pth")
    ap.add_argument("-l", "--label-path", required=False,
        help="SSD labels", type=str, default="./models/ufpark-model-labels.txt")
    ap.add_argument("-s", "--split-point", required=False,
        help="split points on SSD: 0, 1, 2 or 3 (0 means the whole network in receiver)", type=int, default=0)
    args = vars(ap.parse_args())

shapes = {1:(150,150,64), 2:(75,75,128), 3:(38,38,256)}

imageHub = imagezmq.ImageHub()

lastActive = {}
lastActiveCheck = datetime.now

print("# Preparing the server model.")

times = []

class_names = [name.strip() for name in open(args["label_path"]).readlines()]

print("# Creating ssd model.")
model = create_vgg_ssd(len(class_names), is_test=True)

model.load(args["model_path"])
if args["split_point"] != 0:
    _, model2 = create_vgg_ssd_new(len(class_names), model, split=args["split_point"], is_test=True)

    predictor = PredictorM2(
        model2,
        nms_method=None,
        iou_threshold=config.iou_threshold,
        candidate_size=200,
        device=None,
    )
else:
    predictor = create_vgg_ssd_predictor(model, candidate_size=200)
print("# Done!")

# If server and client run in same local directory,
# need a separate place to store the uploads.
try:
    os.mkdir("uploads")
except FileExistsError:
    pass

# start looping over all the frames
times = []

while True:
    if args["split_point"] != 0:
        (rpiName, jpg_buffer) = imageHub.recv_image()
        jpg_buffer = torch.tensor(jpg_buffer).cuda()
        jpg_buffer = jpg_buffer.view(*shapes[args["split_point"]]).permute(2,0,1).unsqueeze(0)
        boxes, labels, probs = predictor.predict(jpg_buffer, 150, 150, 30, 0.4)
    else:
        (rpiName, jpg_buffer) = imageHub.recv_image()
        image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype="uint8"), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, 30, 0.4)
    imageHub.send_reply(b'OK')
    times.append(time.time())
    
    if len(times) > 1:
        print("FPS = {}".format(1./(times[-1] - times[-2])))
