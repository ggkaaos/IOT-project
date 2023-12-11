# USAGE
# python publisher.py --server-ip SERVER_IP

import base64
import imutils
import argparse
import time
import zmq
import cv2
import socket

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True, help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://{}:5566".format(args["server_ip"]))

rpi_name = socket.gethostname()
rpi_name_bytes = rpi_name.encode('utf-8')

vid = cv2.VideoCapture(0)

while True:
    _, frame = vid.read()
    frame = imutils.resize(frame, width=640)
    encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = base64.b64encode(buffer)
    message = (data, rpi_name_bytes)
    pub_socket.send_multipart(message)
    cv2.imshow("publisher image", frame)
    key = cv2.waitKey(1) & 0xFF
    time.sleep(0.01)
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
