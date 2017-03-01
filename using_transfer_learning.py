import os
import glob
import h5py
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from vgg16 import VGG16
from keras import optimizers

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

top_model_weights_path = 'bottleneck_fc_model.h5'
print(K.image_dim_ordering())
img_width, img_height = 150, 150
train_data_dir = 'data/trains'
validation_data_dir = 'data/validations'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=4,
        class_mode='binary')

nb_train_samples = 2000
nb_validation_samples = 800

def load_models():
	model_vgg = Sequential()
	model_vgg.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3)))
	model_vgg.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
	model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
	model_vgg.add(ZeroPadding2D((1, 1)))
	model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
	model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

	f = h5py.File('vgg16_weights.h5')
	for k in range(f.attrs['nb_layers']):
		if k >= len(model_vgg.layers) - 1:
		# we don't look at the last two layers in the savefile (fully-connected and activation)
			break
		g = f['layer_{}'.format(k)]
		weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
		layer = model_vgg.layers[k]

		if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
			weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

		layer.set_weights(weights)
	f.close()
	model_vgg.summary()
	print("VGG16 Model with No Top loaded...")

	top_model = Sequential()
	top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(1, activation='sigmoid'))
	top_model.load_weights(top_model_weights_path)
	top_model.summary()
	print("Top Model loaded...")

	model_vgg.add(top_model)
	model_vgg.summary()
	print("model <- top_model...")
	for layer in model_vgg.layers[:25]:
		layer.trainable = False
	return model_vgg

def test_predict(model, img):
	pass

if __name__ == '__main__':
	model = load_models()
	model.compile(loss='binary_crossentropy',
          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
          metrics=['accuracy'])

	img_path = 'data/test1/14.jpg'
	img = image.load_img(img_path, target_size=(150, 150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features = model.predict(x)
	print(features)
	
	ret = model.evaluate_generator(validation_generator, nb_validation_samples)
	print(ret)
	import gc; gc.collect()
	K.clear_session()
