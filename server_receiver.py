from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.vgg_ssd_new import create_vgg_ssd_new
from vision.ssd.predictor import PredictorM1, PredictorM2
from vision.ssd.config import vgg_ssd_config as config
import socket
import sys
import json
import torch

print("# Preparing the server model.")

model_path = "./models/ufpark-model.pth"
split = 0   # Split point int the first maxpooling layer.
label_path = "./models/ufpark-model-labels.txt"

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

print("# Preparing server socket.")

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('0.0.0.0', 10000)
sock.bind(server_address)

print("# Done!")

print("# Listen for incoming connections.")

# Listen for incoming connections
sock.listen(1)

filename = "data.json"
BUFFER_SIZE = 4096


while True:
    # Wait for a connection
    connection, client_address = sock.accept()

    try:
        with open(filename, "wb") as f:
            while True:
                bytes_read = connection.recv(BUFFER_SIZE)
                if not bytes_read:
                    break
                # write to the file the bytes we just received
                f.write(bytes_read)
                
        with open(filename) as json_file:
            input_batch, height, width = json.load(json_file).values()
        input_batch = torch.tensor(input_batch).cuda()
        boxes, labels, probs = predictor_m2.predict(input_batch, height, width, 30, 0.4)
        print(boxes.shape, labels.shape, probs.shape) 
    finally:
        # Clean up the connection
        print('# Closing socket')
        connection.close()
        print("# Done!")
        break
