# Plot ad hoc OS data instances
from utils import load_labelled_patches, load_class_names
import matplotlib.pyplot as plt
from keras import backend as K

# load 4 randomly selected 128x128 patches and their labels
(X, y) = load_labelled_patches(["SU4010"], 128, limit=4, shuffle=True)

# load the list of possible labels
clznames = load_class_names()

#if we're using the theano backend, we need to change indexing order for matplotlib to interpret the patches:
if K.image_dim_ordering() == 'th':
	X = X.transpose(0, 3, 1, 2)

# plot 4 images
plt.subplot(221).set_title(clznames[y[0].argmax()])
plt.imshow(X[0])
plt.subplot(222).set_title(clznames[y[1].argmax()])
plt.imshow(X[1])
plt.subplot(223).set_title(clznames[y[2].argmax()])
plt.imshow(X[2])
plt.subplot(224).set_title(clznames[y[3].argmax()])
plt.imshow(X[3])

# show the plot
plt.show()