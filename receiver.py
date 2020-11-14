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
        help="split points on SSD: 0, 1, 2, 3 or 4", type=str, default=0)
    args = vars(ap.parse_args())

imageHub = imagezmq.ImageHub()

lastActive = {}
lastActiveCheck = datetime.now

print("# Preparing the server model.")

times = []

class_names = [name.strip() for name in open(args["label_path"]).readlines()]

print("# Creating ssd model.")
model = create_vgg_ssd(len(class_names), is_test=True)

model.load(args["model_path"])
_, model2 = create_vgg_ssd_new(len(class_names), model, split=args["split_point"], is_test=True)

predictor_m2 = PredictorM2(
    model2,
    nms_method=None,
    iou_threshold=config.iou_threshold,
    candidate_size=200,
    device=None,
)
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
    (rpiName, jpg_buffer) = imageHub.recv_image()
    # image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=float), -1)
    # print(np.array(jpg_buffer.shape))
    input_batch = torch.tensor(jpg_buffer).cuda()
    boxes, labels, probs = predictor_m2.predict(input_batch, 150, 150, 30, 0.4)
    imageHub.send_reply(b'OK')
    times.append(time.time())
    
    if len(times) > 1:
        print(1./(times[-1] - times[-2]))



########################################################################################

# s = socket.socket()
# s.bind((HOST, PORT))
# s.listen(10)
# print("Waiting for a connection.....")

# while True:
#     conn, addr = s.accept()
#     print("Got a connection from ", addr)
#     connbuf = buffer.Buffer(conn)

#     while True:
#         # hash_type = connbuf.get_utf8()
#         # if not hash_type:
#         #     break
#         # print('hash type: ', hash_type)

#         file_name = connbuf.get_utf8()
#         if not file_name:
#             break
#         file_name = os.path.join("uploads", file_name)
#         print("file name: ", file_name)

#         file_size = int(connbuf.get_utf8())
#         print("file size: ", file_size)

#         with open(file_name, "wb") as f:
#             remaining = file_size
#             while remaining:
#                 chunk_size = 4096 if remaining >= 4096 else remaining
#                 chunk = connbuf.get_bytes(chunk_size)
#                 if not chunk:
#                     break
#                 f.write(chunk)
#                 remaining -= len(chunk)
#             # if remaining:
#             #     print('File incomplete.  Missing',remaining,'bytes.')
#             # else:
#             #     print('File received successfully.')
#         with open(file_name) as json_file:
#             input_batch, height, width = json.load(json_file).values()
#             input_batch = torch.tensor(input_batch).cuda()
#             boxes, labels, probs = predictor_m2.predict(input_batch, height, width, 30, 0.4)
#             times.append(time.time())
#     print("Connection closed.")
#     conn.close()
#     break

# output = [times[i] - times[i - 1] for i in range(1, len(times))]
# print(output)