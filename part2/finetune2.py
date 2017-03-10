from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from utils import generate_labelled_patches, load_labelled_patches, load_class_names
from keras import optimizers

# load the class names
clznames = load_class_names()
num_classes = len(clznames)

def hack_resnet(num_classes):
	model = ResNet50(include_top=True, weights='imagenet')

	# Get input
	new_input = model.input
	# Find the layer to connect
	hidden_layer = model.layers[-2].output
	# Connect a new layer on it
	new_output = Dense(num_classes) (hidden_layer)
	# Build a new model
	newmodel = Model(new_input, new_output)

	return newmodel

model = hack_resnet(num_classes)

# the resnet expects 224x224 inputs
patch_size = 224

# load data
train_data = generate_labelled_patches(["SU4010"], patch_size, shuffle=True)
valid_data = load_labelled_patches(["SU4011"], patch_size, limit=1000, shuffle=True)

# set weights in all but last layer
# to non-trainable (weights will not be updated)
for layer in model.layers[:len(model.layers)-2]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# fine-tune the model
model.fit_generator(
        train_data,
        samples_per_epoch=10016,
        nb_epoch=10,
        validation_data=valid_data)