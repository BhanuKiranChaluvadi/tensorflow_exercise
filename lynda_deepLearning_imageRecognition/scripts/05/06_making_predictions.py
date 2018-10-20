from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

# Load the json file that contains the model's structure
model_structure = open("model_structure.json", "rb").read()

# Recreate the Keras model object from the json file
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_weights.h5")

# Load an image file to test.
# Resize it to  64X64 pixels (as required by this model)
img = image.load_img("not_dog.png", target_size=(64, 64))

# Convert the image to a numpy array
image_array = image.img_to_array(img)

# Add a fourth dimension to the image (since Keras expect a bunch of images, not a single image)
images = np.expand_dims(image_array, axis=0)

# Normalize the data
images = vgg16.preprocess_input(images)

# Use the pre-trained neural network to extract feature from our test image (the same way we did to train the model)
feature_extraction_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
features = feature_extraction_model.predict(images)
# print "Feature: ", features

# Make a final prediction using our model
results = model.predict(features)

# Check the first result first element.
single_result = results[0][0]

# Print the result
print("Likelihood that this image contains a dog: {}%".format(int(single_result * 100)))

