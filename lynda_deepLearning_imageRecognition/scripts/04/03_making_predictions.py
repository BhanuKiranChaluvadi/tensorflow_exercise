from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# List of names for each CIFAR10 class in order 0 o 9
class_labels = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck",
]


# Load the json file that contains the model's structure
model_structure = open("model_structure.json", "rb").read()

# Recreate the Keras model object from the json file
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights('model_weights.h5')

# Load an image file to test, resize it to 32X32 pixels(as required by this model)
img = image.load_img("car.png", target_size=(32, 32))

# Convert the image to a numpy array
image_to_test = image.img_to_array(img)

# Add a fourth dimension to the image(since Keras expect a list of images, not a single image)
list_of_images = np.expand_dims(image_to_test, axis=0)

# Make a prediction using the model
results = model.predict(list_of_images)

# Since we are only testing one image, we only need to check the first result
single_result = results[0]

# We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

# Get the name of the most likely class
class_label = class_labels[most_likely_class_index]

# Print the result
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))