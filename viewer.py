# USAGE
# python viewer.py --montageW 2 --montageH 2

import argparse
from datetime import datetime
import imutils
import zmq
import cv2
import numpy as np
from imutils import build_montages

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-mW", "--montageW", required=True, type=int, help="montage frame width")
ap.add_argument("-mH", "--montageH", required=True, type=int, help="montage frame height")
args = vars(ap.parse_args())

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

# assign montage width and height, so we can view all incoming frames in a single "dashboard"
mW = args["montageW"]
mH = args["montageH"]

frame_dict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
last_active = {}
last_active_check = datetime.now()

context = zmq.Context()
view_socket = context.socket(zmq.SUB)
view_socket.connect("tcp://localhost:5577")
view_socket.setsockopt_string(zmq.SUBSCRIBE, "")

while True:
    message = view_socket.recv_multipart()
    rpi_name_bytes, img_data = message
    rpi_name = rpi_name_bytes.decode('utf-8')

    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # if a device is not in the last active dictionary then it means
    # that it's a newly connected device
    if rpi_name not in last_active.keys():
        print("[INFO] receiving data from {}...".format(rpi_name))

    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    # record the last active time for the device from which we just
    # received a frame
    last_active[rpi_name] = datetime.now()

    cv2.putText(frame, rpi_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # label = ", ".join("{}: {}".format(obj, count) for (obj, count) in obj_count.items())
    # cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame_dict[rpi_name] = frame

    # build a montage using images in the frame dictionary
    montages = build_montages(frame_dict.values(), (w, h), (mW, mH))
    # display the montage(s) on the screen
    for (i, montage) in enumerate(montages):
        cv2.imshow("Viewer ({})".format(i), montage)

    # time.sleep(0.01)

    key = cv2.waitKey(1) & 0xFF

    # if current time *minus* last time when the active device check
    # was made is greater than the threshold set then do a check
    if (datetime.now() - last_active_check).seconds > ACTIVE_CHECK_SECONDS:
        # loop over all previously active devices
        for (rpiName, ts) in list(last_active.items()):
            # remove the RPi from the last active and frame
            # dictionaries if the device hasn't been active recently
            if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(rpiName))
                last_active.pop(rpiName)
                frame_dict.pop(rpiName)

        # set the last active check time as current time
        lastActiveCheck = datetime.now()

    if key == ord('q'):
        break
cv2.destroyAllWindows()
