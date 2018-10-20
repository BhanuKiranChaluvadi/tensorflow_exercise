from pathlib import Path
import glob, os
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16

# Path to folder with training data
curr_dir = os.path.dirname(__file__)
dog_path = os.path.join(curr_dir, 'training_data/dogs')
not_dog_path = os.path.join(curr_dir, 'training_data/not_dogs')

images = []
labels = []

print "dog path: ", dog_path

# Load all the not-dog images
for img in os.listdir(not_dog_path):
    if img.endswith(".png"):
        # load the image from the disk
        img = image.load_img(os.path.join(not_dog_path, img))

        # convert image to a numpy array
        image_array = image.img_to_array(img)

        # Add the image to the list of images
        images.append(image_array)

        # For each 'not dog' image, the expected value should be 0
        labels.append(0)

# Load all the dog images
for img in os.listdir(dog_path):
    if img.endswith(".png"):
        # load the image from the desk
        img = image.load_img(os.path.join(dog_path, img))

        # convert the image into numpy array
        image_array = image.img_to_array(img)

        # add the image to the list
        images.append(image_array)

        # For eah 'dog' image, the expected value should be 1
        labels.append(1)


print "labels: ", labels

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Normalize the image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as feature extractor
pretrained_nn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

# Extract feature for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file.
joblib.dump(y_train, "y_train.dat")
