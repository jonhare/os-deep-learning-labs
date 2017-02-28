import numpy
import random
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')

from utils import generate_labelled_patches, load_class_mapping
import models

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
random.seed(seed)

patch_size = 28

# load data
train_data = generate_labelled_patches(["SU4111"], patch_size)
valid_data = generate_labelled_patches(["SU4211"], patch_size)

# better way to do this?
mapping = load_class_mapping()
num_classes = len(mapping)

# Compile model
model = models.simple_cnn((3, patch_size, patch_size), num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit_generator(train_data, samples_per_epoch=32000, nb_epoch=10, validation_data=valid_data, nb_val_samples=3200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
