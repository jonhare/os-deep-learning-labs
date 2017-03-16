# Plot ad hoc OS data instances
from utils import load_labelled_patches, load_class_names
import matplotlib.pyplot as plt
from keras import backend as K

# load 4 randomly selected 128x128 patches and their labels
(X, y) = load_labelled_patches(["SU4010"], 128, limit=25, shuffle=True)

# load the list of possible labels
clznames = load_class_names()

#if we're using the theano backend, we need to change indexing order for matplotlib to interpret the patches:
if K.image_dim_ordering() == 'th':
	X = X.transpose(0, 3, 1, 2)

# plot 4 images
for i in xrange(0,4):
	plt.subplot(2,2,i+1).set_title(clznames[y[i].argmax()])
	plt.imshow(X[i])

# show the plot
plt.show()