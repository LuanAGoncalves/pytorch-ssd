from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.vgg_ssd_new import create_vgg_ssd_new
from vision.ssd.predictor import PredictorM1, PredictorM2
from vision.ssd.config import vgg_ssd_config as config
import sys
import cv2
import json
import socket
import threading
import os
import imagezmq
import argparse
import cv2
import time 

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server-ip", required=True,
        help="ip address of the server to which the client will connect")
    ap.add_argument("-v", "--video-file", required=True,
        help="path to a video file", type=str)
    ap.add_argument("-q", "--jpeg-quality", required=False,
        help="0 to 100, higher is better quality, 95 is cv2 default", type=int, default=95)
    ap.add_argument("-m", "--model-path", required=False,
        help="path to the trained model", type=str, default="./models/ufpark-model.pth")
    ap.add_argument("-l", "--label-path", required=False,
        help="SSD labels", type=str, default="./models/ufpark-model-labels.txt")
    ap.add_argument("-p", "--split-point", required=False,
        help="split points on SSD: 0, 1, 2 or 3  (0 means the whole network in receiver)", type=int, default=0)
    args = vars(ap.parse_args())

    class_names = [name.strip() for name in open(args["label_path"]).readlines()]

    shapes = {1:(4000,800,3), 2:(400,600,3), 3:(608,608)}

    print("# Creating ssd model.")
    model = create_vgg_ssd(len(class_names), is_test=True)
    model.load(args["model_path"])

    if args["split_point"] != 0:
        model.load(args["model_path"])
        model1, _ = create_vgg_ssd_new(len(class_names), model, split=args["split_point"], is_test=True)
        predictor = PredictorM1(model1, config.image_size, config.image_mean, device=None,)
    else:
        pass
    print("# Done!")

    sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(args["server_ip"]))
    rpiName = socket.gethostname()
    cap = cv2.VideoCapture(args["video_file"])
    time.sleep(2.0)

    while True:
        # read the frame from the camera and send it to the server
        # frame = vs.read()
        ret, frame = cap.read()
        if ret == True:
            if args["split_point"] != 0:
                output = predictor.predict(frame)
                output = output.view(*shapes[args["split_point"]])
                output = output.cpu().numpy()
                ret_code, output = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), args["jpeg_quality"]])
                sender.send_image(rpiName, output)
            else:
                ret_code, jpg_buffer = cv2.imencode(".jpg", cv2.resize(frame, (300,300), interpolation = cv2.INTER_AREA), [int(cv2.IMWRITE_JPEG_QUALITY), args["jpeg_quality"]])
                sender.send_image(rpiName, jpg_buffer)
        else:
            break