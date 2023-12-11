# USAGE
# python processor.py -p MobileNetSSD_deploy.prototxt -m MobileNetSSD_deploy.caffemodel

import argparse
import base64
import cv2
import imutils
import numpy as np
import zmq

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
# ap.add_argument("-a", "--addresses", nargs='+', required=True, help="list of addresses to connect to")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]
# initialize the consider set (class labels we care about and want to count),
# the object count dictionary, and the frame  dictionary
CONSIDER = {"person", "bottle", "chair", "sofa"}

print("[INFO] detecting: {}...".format(", ".join(obj for obj in CONSIDER)))

objCount = {obj: 0 for obj in CONSIDER}
frame_dict = {}

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

context = zmq.Context()

# SUB socket for receiving from publishers
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://127.0.0.1:5566")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, optval='')

# PUB socket for sending to Viewers
processor_pub = context.socket(zmq.PUB)
processor_pub.bind("tcp://*:5577")

while True:
    message = sub_socket.recv_multipart()
    data, rpi_name_bytes = message
    img = base64.b64decode(data)
    np_img = np.frombuffer(img, dtype=np.uint8)
    source = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rpi_name = rpi_name_bytes.decode('utf-8')

    frame = imutils.resize(source, width=400)
    (h, w) = frame.shape[:2]

    # construct a blob from the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    # reset the object count for each object in the CONSIDER set
    obj_count = {obj: 0 for obj in CONSIDER}

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])

            # check to see if the predicted class is in the set of classes that need to be considered
            if CLASSES[idx] in CONSIDER:
                # increment the count of the particular object detected in the frame
                obj_count[CLASSES[idx]] += 1
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the bounding box around the detected object on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # draw the object count on the frame
    cv2.putText(frame, rpi_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in obj_count.items())
    cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame_dict[rpi_name] = frame

    processor_pub.send_multipart([rpi_name.encode('utf-8'), cv2.imencode('.png', frame)[1].tobytes()])

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
cv2.destroyAllWindows()
