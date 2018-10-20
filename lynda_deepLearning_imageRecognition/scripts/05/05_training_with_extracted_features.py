from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib

# Load data set
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

# Create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# print(model.summary())

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
with open('model_structure.json', "w") as json_file:
    json_file.write(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")
