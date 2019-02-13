# face-detection

Generating named bounding boxes on known/unknown faces

## Requirements 

This script requires Python 3.6+ 

**Install dependencies via pip:**

```
pip install -r requirements.txt
``` 
#### Virtualenv

**Step 1: Creating new virtual environment**

Best practice is to set up development environment using  ```virtualenv``` package

You can install it via pip:
```
pip install virtualenv
```

```
python -m venv venv
```
or

```
virtualenv -p python venv
```

**Step 2: Activate virtual environment**

For Linux and Mac:
```
source venv/bin/activate
```
For Windows:
```
venv\Scripts\activate.bat
```

**Deactivate the environment with**

```
deactivate
```

**(Additional)**
 Use: `python -h venv` for help

For more details about how to set up your virtual environment, [read the Docs](https://docs.python.org/3/library/venv.html)



## Usage

Let the to NN detect and store known faces data:

      
      python encode_faces.py -e ENCODINGS_PICKLE_PATH -c cnn
          
      Options:
          -d --dataset Path to folder with dataset
          (May contain subfolders, each named as a concrete person you want to recognize)
          
          -e --encodings Path to .pickle file where to serialize face encodings
          -m --method Learning model: use 'cnn' or 'hog'
           [
             'hog' : (Histogram of oriented gradients) -> faster, less accurate
             'cnn' : (Convolutional Neural Network) -> slower, more accurate
           ]

Run the real-time face recognition (Web-camera is required):

      python pi_face_recognition.py -e ENCODINGS_PICKLE_PATH -c CASCADE_PATH
          
      Options:
          -e --encodings Path to .pickle file with known face encodings
          -c --cascade Path to .xml cascade classifier file