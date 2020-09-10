from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.vgg_ssd_new import create_vgg_ssd_new
from vision.ssd.predictor import PredictorM1, PredictorM2
from vision.ssd.config import vgg_ssd_config as config
import socket
import sys
import cv2
import json

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

print("# Preparing client socket.")

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('200.239.93.228', 10000)
sock.connect(server_address)

print("# Done!")

print("# Start processing!")

orig_image = cv2.imread('../dataset/JPEGImages/afternoon_1_scene_0015001.jpg')
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape
output = predictor_m1.predict(image)

output = output.cpu().numpy().tolist()

filename = "data.json"
BUFFER_SIZE = 4096

with open(filename, "w") as outfile:
    json.dump({"scores": output, "height":height, "width":width}, outfile)

print("# Done!")

print("# Sending scores.")
try:
    
    # Send data
    # message = 'This is the message.  It will be repeated.'
    # print >>sys.stderr, 'sending "%s"' % message
    # sock.sendall(message)

    # with open(filename, "rb") as f:
    #     bytes_read = f.read()
    #     sock.sendall(bytes_read)

    with open(filename, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in
            # busy networks
            sock.sendall(bytes_read)
    # # Look for the response
    # amount_received = 0
    # amount_expected = len(message)
    
    # while amount_received < amount_expected:
    #     data = sock.recv(16)
    #     amount_received += len(data)
    #     print >>sys.stderr, 'received "%s"' % data

finally:
    print('Closing socket')
    sock.close()