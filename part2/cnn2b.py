import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pylab as plt
from utils import generate_labelled_patches, load_labelled_patches, load_class_names
from keras import backend as K

# define the patch size as a variable so its easier to change later. For now,
# we'll set it to 28, just like the mnist images
patch_size = 28

# load data
train_data = generate_labelled_patches(["SU4010"], patch_size, shuffle=True)
valid_data = load_labelled_patches(["SU4011"], patch_size, subcoords=((0,0), (300,300)))

# load the class names
clznames = load_class_names()
num_classes = len(clznames)

class DisplayMap(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		clzs = model.predict_classes(valid_data[0])
		clzs = clzs.reshape((300-patch_size, 300-patch_size))
		plt.figure()
		plt.imshow(clzs)
		plt.savefig("map_epoch%s.png" % epoch)

def larger_model(input_shape, num_classes):
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

# build the model
model = larger_model(valid_data[0][0].shape, num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit_generator(train_data, samples_per_epoch=10016, nb_epoch=10, validation_data=valid_data, verbose=1, callbacks=[DisplayMap()])

# load some test data; this time we specifically load patches from a 300x300 square of a tile in scan-order
test_data = load_labelled_patches(["SU4012"], patch_size, subcoords=((0,0), (300,300)))

# we can reshape the test data labels back into an image and save it
tmp = np.zeros(test_data[1].shape[0])
for x in xrange(0, test_data[1].shape[0]):
	tmp[x] = test_data[1][x].argmax()
tmp = tmp.reshape((300-patch_size, 300-patch_size))
plt.figure()
plt.imshow(tmp)
plt.savefig("test_gt.png")

# and we can do the same for the predictions
clzs = model.predict_classes(test_data[0])
clzs = clzs.reshape((300-patch_size, 300-patch_size))
plt.figure()
plt.imshow(clzs)
plt.savefig("test_pred.png")
