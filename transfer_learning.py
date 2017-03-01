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

weights_path = './vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

# train_data_dir = '../own-object-detector/dataset/CarData/Trains'
# validation_data_dir = '../own-object-detector/dataset/CarData/Validations'
train_data_dir = 'data/trains/'
validation_data_dir = 'data/validations'

nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 30

TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

def save_bottlebeck_features():
	datagen = ImageDataGenerator(rescale=1./255)

	model = VGG16(include_top=False, weights='imagenet')
	print("Empty Top Model Loaded...")

	# print('Model loaded.')

	generator = datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=16,
		class_mode=None,
		shuffle=False)

	bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
	np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
	print("bottleneck_train.npy is created..")

	generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=4,
            class_mode=None,
            shuffle=False)

	bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
	np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
	print("bottleneck_validation.npy is created..")

def train_top_model():
	train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
	# 0: cat, 1: dog
	train_labels = np.array([0]*(nb_train_samples//2) + [1]*(nb_train_samples//2))
	
	validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
	validation_labels = np.array([0]*(nb_validation_samples//2) + [1]*(nb_validation_samples//2))
	
	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(train_data, train_labels,
	          nb_epoch=nb_epoch, batch_size=16,
	          validation_data=(validation_data, validation_labels),
	          verbose=1)
	model.save_weights(top_model_weights_path)
	print("Bottle neck weights are saved...")

if __name__ == '__main__':
	save_bottlebeck_features()
	train_top_model()
	import gc; gc.collect()
	K.clear_session()



