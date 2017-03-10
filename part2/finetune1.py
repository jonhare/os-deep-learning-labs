from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input
import numpy as np

model = ResNet50(include_top=True, weights='imagenet')

img_path = 'images/mf.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
