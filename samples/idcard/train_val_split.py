#
import os
import sys
import glob as glob
import numpy as np
import shutil

from sklearn.model_selection import train_test_split

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

def maybe_makedir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

data_folder = os.path.join(ROOT_DIR, 'datasets/back_bbox_json/back_bbox')
obj_folder = os.path.join(ROOT_DIR, 'datasets/idcard')

images = []
labels = []
for i in range(10):
	images_path = os.path.join(data_folder, 'back{}'.format(i))
	_images = glob.glob('{}/*.jpg'.format(images_path))
	for _image in _images:
		images.append(_image)
		name = os.path.splitext(os.path.basename(_image))[0]
		labels.append(os.path.join(data_folder, 'json_outputs', name + '.json'))

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)

maybe_makedir(os.path.join(obj_folder, 'train', 'images'))
maybe_makedir(os.path.join(obj_folder, 'train', 'labels'))
maybe_makedir(os.path.join(obj_folder, 'val', 'images'))
maybe_makedir(os.path.join(obj_folder, 'val', 'labels'))

for _x_train, _y_train in zip(X_train, y_train):
	shutil.copy(_x_train, os.path.join(obj_folder, 'train', 'images'))
	shutil.copy(_y_train, os.path.join(obj_folder, 'train', 'labels'))
for _x_val, _y_val in zip(X_val, y_val):
	shutil.copy(_x_val, os.path.join(obj_folder, 'val', 'images'))
	shutil.copy(_y_val, os.path.join(obj_folder, 'val', 'labels'))