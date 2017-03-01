import glob
import os
from shutil import copyfile, rmtree
train_img_path = 'data/train/'
train_dir_path = 'data/trains/'
validation_dir_path = 'data/validations/'
pos_img_cnt = 0 
neg_img_cnt = 0

pos_train_dir_path = train_dir_path + 'dog'
neg_train_dir_path = train_dir_path + 'cat'

pos_validation_dir_path = validation_dir_path + 'dog'
neg_validation_dir_path = validation_dir_path + 'cat'

dir_paths = [pos_train_dir_path, neg_train_dir_path, 
			pos_validation_dir_path, neg_validation_dir_path]

nb_train_samples = 1000
nb_validation_samples = 400
nb_epoch = 50

for each_path in dir_paths:
	if os.path.isdir(each_path):
		rmtree(each_path)
		os.makedirs(each_path)
		print("Clean %s directory.." % each_path)
	elif not os.path.isdir(each_path):
		os.makedirs(each_path)
		print("Making %s directory.." % each_path)	

print(os.listdir(train_dir_path))

for im_path in glob.glob(os.path.join(train_img_path, "*")):
	img_name = im_path.split('/')[-1]
	# print(img_name)
	class_ = img_name.split('.')[0]
	if class_ == 'dog':
		if pos_img_cnt < nb_train_samples:
			copyfile(im_path, pos_train_dir_path+'/'+img_name)
		else:
			if pos_img_cnt >= nb_train_samples + nb_validation_samples:
				continue
			copyfile(im_path, pos_validation_dir_path+'/'+img_name)
		pos_img_cnt += 1
	elif class_ == 'cat':
		if neg_img_cnt < nb_train_samples:
			copyfile(im_path, neg_train_dir_path+'/'+img_name)
		else:
			if neg_img_cnt >= nb_train_samples + nb_validation_samples:
				continue
			copyfile(im_path, neg_validation_dir_path + '/'+ img_name)
		neg_img_cnt += 1

print(len(os.listdir(os.path.join(train_dir_path, 'dog'))))
print(len(os.listdir(os.path.join(train_dir_path, 'cat'))))
print(len(os.listdir(os.path.join(validation_dir_path, 'dog'))))
print(len(os.listdir(os.path.join(validation_dir_path, 'cat'))))

print("dog cnt ", pos_img_cnt)
print("cat cnt ", neg_img_cnt)