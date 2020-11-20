#Produced by a comp sci UBD student with id 15B4012

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import warnings
import json
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-con", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# filter warnings, load the configuration
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.5)
fps = FPS().start()
avg = None
lastUploaded = datetime.datetime.now()
motion_counter = 0
non_motion_timer = conf["nonMotionTimer"]
fourcc = 0x00000021  # for linux distrubtion
writer = None
(w, h) = (None, None)
zeros = None
output = None
made_recording = False

# loop over the frames from the video stream
while True:

	timestamp = datetime.datetime.now()
	motion_detected = False

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			motion_detected = True

		# Check if writer is None TODO: make path configurable
		if writer is None:
			#fourcc = cv2.VideoWriter_fourcc(*'FMP4')
			#fourcc = cv2.VideoWriter_fourcc(*'XVID')
			#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
			#fourcc = cv2.VideoWriter_fourcc(*'MPEG')
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
			file_path = ("/home/pi/Downloads/realtime/output/video/{filename}.avi")
			file_path = file_path.format(filename=filename)

			(h2, w2) = frame.shape[:2]
			writer = cv2.VideoWriter(file_path, fourcc, 3, (w2, h2), True)
			zeros = np.zeros((h2, w2), dtype="uint8")

		def record_video():
			# construct the final output frame, storing the original frame
			output = np.zeros((h2, w2, 3), dtype="uint8")
			output[0:h2, 0:w2] = frame

			# write the output frame to file
			writer.write(output)
			#print("[INFO] Recording....")

		if motion_detected:
			# increment the motion counter
			motion_counter += 1
			# check to see if the number of frames with motion is high enough
			if motion_counter >= conf["min_motion_frames"]:
				if conf["create_image"]:
					# create image TODO: make path configurable
					image_path = ("/home/pi/Downloads/realtime/output/image/{filename}.jpg").format(filename=filename)
					cv2.imwrite(image_path, frame)
				
				record_video()
				made_recording = True
				non_motion_timer = conf["nonMotionTimer"]

		# If there is no motion, continue recording until timer reaches 0
		# Else clean everything up
		else:  # TODO: implement a max recording time
			# print("[DEBUG] no motion")
			if made_recording is True and non_motion_timer > 0:
				non_motion_timer -= 1
				# print("[DEBUG] first else and timer: " + str(non_motion_timer))
				record_video()

			else:
            			# print("[DEBUG] hit else")
				motion_counter = 0
				if writer is not None:
					# print("[DEBUG] hit if 1")
					writer.release()
					writer = None

				if made_recording is False:
					# print("[DEBUG] hit if 2")
					os.remove(file_path)
					made_recording = False
					non_motion_timer = conf["nonMotionTimer"]

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# clean up
cv2.destroyAllWindows()
vs.stop()
#writer.release()
