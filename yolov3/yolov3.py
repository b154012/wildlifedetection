import cv2
import numpy as np
from imutils.video import FPS

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# initialize the camera and grab a reference to the raw camera capture
#initialize the FPS counter
print("[INFO] starting video stream...")
vs= cv2.VideoCapture(0)
fps = FPS().start()

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")

while True:
	_,frame = vs.read()
	height, width, channels = frame.shape

	# Detecting objects
	blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

		font = cv2.FONT_HERSHEY_PLAIN

		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				color = colors[i]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
	cv2.imshow("WildLife Detector", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

	fps.update()


#stop fps
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.release()
cv2.destroyAllWindows()
