from imutils import paths
import cv2
import face_recognition
import argparse
import pickle
import os

# construct cli agrument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
				help="path to the image directory")
ap.add_argument("-e", "--encodings", required=True,
				help="path to the .pickle facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
				help="face detection model to use: either `hog` for Raspberry Pi or `cnn` for pc's")
args = vars(ap.parse_args())

# grab the paths to the input images in dataset folder
print("[INFO] quantifying faces...")
image_paths = list(paths.list_images(args["dataset"]))

encodings = []
names = []

for (i, image_path) in enumerate(image_paths):
	# extract the person name from the image path
	print("[INFO] processing image file {}/{}".format(i + 1, len(image_paths)))
													  
	# store the person name for current image set
	name = image_path.split(os.path.sep)[-2]

	# load the image and convert it from BGR to RGB
	print(image_path)
	image = cv2.imread(image_path)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# detect the (x, y)-coordinates of the bounding face boxes
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])		

	# compute the facial encodings for the current face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		encodings.append(encoding)
		names.append(name)

# serialize obtained data 
print("[INFO] serializing face encodings...")
data = {"encodings": encodings, "names": names}
file = open(args["encodings"], "wb")
file.write(pickle.dumps(data, protocol=4))
file.close()
