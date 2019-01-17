#################################################################################################
# Run Command- python image_capture.py -o dataset/<name>
#

from imutils.video import VideoStream
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-c", "--confidence", type=float, default=0.4,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

if not os.path.exists(args["output"]):
	os.makedirs(args["output"])

# load face detector model and prototype
#print("Loading face detector...")
cascadePath = os.path.sep.join(["face_detection_model", "haarcascade_frontalface_default.xml"])
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model","res10_300x300_ssd_iter_140000.caffemodel"])
img_capture = cv2.CascadeClassifier(cascadePath)
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("\nStarting video for CAPTURING IMAGES...\n")
print("Press 'c' to capture images and 'Esc' key to stop\n")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0
while True:
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=800)
 
	# detect faces in the grayscale frame
	rects = img_capture.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,minNeighbors=5, minSize=(50, 30))
 
	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("c"):
		p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

	elif key==27:
		break

print("{} face images stored in location {}\n".format(total, args["output"]))
cv2.destroyAllWindows()
vs.stop()

# load serialized face embedding model
#print("Loading face recognizer...")
embedderPath = os.path.sep.join(["face_detection_model", "openface_nn4.small2.v1.t7"])
embedder = cv2.dnn.readNetFromTorch(embedderPath)

# grab the paths to the input images in our dataset
print("Processing Images...\n")
imagePaths = list(paths.list_images("dataset"))
#print(imagePaths)

# initialize lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	#print("Processing image {}/{}".format(i + 1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=800)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# dump the facial embeddings + names to disk
#print("Serializing {} encodings...".format(total))
print("Done Processing, Run face_detection.py\n")
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
cv2.destroyAllWindows()
vs.stop()