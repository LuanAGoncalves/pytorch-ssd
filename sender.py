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


import buffer

print("# Preparing the client model.")

model_path = "./models/ufpark-model.pth"
split = 0   # Split point int the first maxpooling layer.
label_path = "./models/ufpark-model-labels.txt"

class_names = [name.strip() for name in open(label_path).readlines()]

print("# Creating ssd model.")
model = create_vgg_ssd(len(class_names), is_test=True)

model.load(model_path)
model1, _ = create_vgg_ssd_new(
    len(class_names), model, split=split, is_test=True
)

predictor_m1 = PredictorM1(
        model1,
        config.image_size,
        config.image_mean,
        device=None,
    )
print("# Done!")

HOST = '200.239.93.134'
PORT = 2345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

with s:
    sbuf = buffer.Buffer(s)

    # hash_type = input('Enter hash type: ')

    # files = input('Enter file(s) to send: ')
    # files_to_send = files.split()

    for file_name in os.listdir("images"):
        # print(file_name)
        # sbuf.put_utf8(hash_type)
        sbuf.put_utf8(file_name.split("/")[-1])

        orig_image = cv2.imread('./images/'+file_name)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        output = predictor_m1.predict(image)

        output = output.cpu().numpy().tolist()

        filename = "data.json"
        with open(filename, "w") as outfile:
            json.dump({"scores": output, "height":height, "width":width}, outfile)

        file_size = os.path.getsize(filename)
        sbuf.put_utf8(str(file_size))

        with open(filename, 'rb') as f:
            sbuf.put_bytes(f.read())
        print('File Sent')
        os.remove(filename)