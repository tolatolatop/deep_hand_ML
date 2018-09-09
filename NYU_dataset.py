from __future__ import print_function

#import tensorflow as tf
import os
import re

MODE = "folder"

DATASET_PATH = r".\exm3\train"

RE_LABEL = re.compile("[\\w]+_([\\d])+_([\\d])+")

IMG_HEIGHT = 480
IMG_WIDTH = 640
CHANNELS = 3

def read_images(dataset_path, mode, bathch_size):
	imagepaths, labels = list(), list()
	rgb_imagepaths, synth_imagepaths,depth_imagepaths = list(), list(),list()
	rgb_labels, synth_labels, depth_labels = list() , list() , list()
	if mode == "file":
		return None
	elif mode == "folder":
		samples = os.walk(dataset_path).__next__()[2]
		for sample in samples:
			if '.png' in sample:
				if 'rgb' in sample:
					rgb_imagepaths.append(os.path.join(dataset_path,sample))
					rgb_labels.append([int(i) for i in RE_LABEL.findall(sample)[0]])
				elif 'synth' in sample:
					synth_imagepaths.append(os.path.join(dataset_path,sample))
					synth_labels.append([int(i) for i in RE_LABEL.findall(sample)[0]])
				elif 'depth' in sample:
					depth_imagepaths.append(os.path.join(dataset_path,sample))
					depth_labels.append([int(i) for i in RE_LABEL.findall(sample)[0]])
	else:
		raise Exception("Unknown mode.")
		

read_images(DATASET_PATH,MODE,64)

