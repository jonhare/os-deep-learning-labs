from resnet50 import ResNet50
from imagenet_utils import preprocess_input
from keras.models import Model
from utils import load_labelled_patches


model = ResNet50(include_top=True, weights='imagenet')

# Get input
new_input = model.input

# Find the layer to end on
new_output = model.layers[-2].output

# Build a new model
newmodel = Model(new_input, new_output)

(X, y_test_true) = load_labelled_patches(["SU4012"], 224, limit=4, shuffle=True)
X = preprocess_input(X)

features = newmodel.predict(X)

print features.shape
print features