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

print("# Preparing the server model.")

model_path = "./models/ufpark-model.pth"
split = 0   # Split point int the first maxpooling layer.
label_path = "./models/ufpark-model-labels.txt"
times = []

class_names = [name.strip() for name in open(label_path).readlines()]

print("# Creating ssd model.")
model = create_vgg_ssd(len(class_names), is_test=True)

model.load(model_path)
_, model2 = create_vgg_ssd_new(
    len(class_names), model, split=split, is_test=True
)

predictor_m2 = PredictorM2(
        model2,
        nms_method=None,
        iou_threshold=config.iou_threshold,
        candidate_size=200,
        device=None,
    )
print("# Done!")

HOST = '0.0.0.0'
PORT = 2345

# If server and client run in same local directory,
# need a separate place to store the uploads.
try:
    os.mkdir('uploads')
except FileExistsError:
    pass

s = socket.socket()
s.bind((HOST, PORT))
s.listen(10)
print("Waiting for a connection.....")

while True:
    conn, addr = s.accept()
    print("Got a connection from ", addr)
    connbuf = buffer.Buffer(conn)

    while True:
        # hash_type = connbuf.get_utf8()
        # if not hash_type:
        #     break
        # print('hash type: ', hash_type)

        file_name = connbuf.get_utf8()
        if not file_name:
            break
        file_name = os.path.join('uploads',file_name)
        print('file name: ', file_name)

        file_size = int(connbuf.get_utf8())
        print('file size: ', file_size )

        with open(file_name, 'wb') as f:
            remaining = file_size
            while remaining:
                chunk_size = 4096 if remaining >= 4096 else remaining
                chunk = connbuf.get_bytes(chunk_size)
                if not chunk: break
                f.write(chunk)
                remaining -= len(chunk)
            # if remaining:
            #     print('File incomplete.  Missing',remaining,'bytes.')
            # else:
            #     print('File received successfully.')
        with open(file_name) as json_file:
            input_batch, height, width = json.load(json_file).values()
        input_batch = torch.tensor(input_batch).cuda()
        boxes, labels, probs = predictor_m2.predict(input_batch, height, width, 30, 0.4)
        times.append(time.time())
    print('Connection closed.')
    conn.close()
    break

output = [times[i] - times[i-1] for i in range(1,len(times))]
print(output)