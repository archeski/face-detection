from imutils import paths
import cv2
import face_recognition
import argparse
import pickle
import os


def encode_faces(dataset, encodings_file, detection_method):
    # grab the paths to the input images in dataset folder
    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(dataset))
    known_encodings, known_names = [], []

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
        boxes = face_recognition.face_locations(rgb, model=detection_method)

        # compute the facial encodings for the current face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    # serialize obtained data
    print("[INFO] serializing face encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    file = open(encodings_file, "wb")
    file.write(pickle.dumps(data, protocol=4))
    file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataset", required=True,
                        help="path to the image directory")
    parser.add_argument("-e", "--encodings", required=True,
                        help="path to the .pickle facial encodings")
    parser.add_argument("-m", "--method", type=str, default="cnn",
                        help="face detection model to use: either `hog` for Raspberry Pi or `cnn` for pc's")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    encode_faces(args.dataset, args.encodings, args.method)
