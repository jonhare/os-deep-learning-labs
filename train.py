import numpy as np
import keras
import random
import matplotlib.pylab as plt
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')

from utils import generate_labelled_patches, load_class_mapping, load_labelled_patches
import models

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
random.seed(seed)

patch_size = 28

# load data
train_data = generate_labelled_patches(["SU4111"], patch_size, shuffle=True)
valid_data = load_labelled_patches(["SU4211"], patch_size, subcoords=((0,0), (300,300)))

tmp = np.zeros(valid_data[1].shape[0])
for x in xrange(0, valid_data[1].shape[0]):
	tmp[x] = valid_data[1][x].argmax()
tmp = tmp.reshape((272,272))
plt.figure()
plt.imshow(tmp)
plt.savefig("map_gt.png")

# better way to do this?
mapping = load_class_mapping()
num_classes = len(mapping)

# Compile model
model = models.simple_cnn((3, patch_size, patch_size), num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class DisplayMap(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		clzs = model.predict_classes(valid_data[0])
		clzs = clzs.reshape((272,272))
		plt.figure()
		plt.imshow(clzs)
		plt.savefig("map_epoch%s.png" % epoch)

# Fit the model
model.fit_generator(train_data, samples_per_epoch=64000, nb_epoch=10, validation_data=valid_data, verbose=1, callbacks=[DisplayMap()])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
