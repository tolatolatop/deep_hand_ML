from __future__ import print_function

import tensorflow as tf
import os
import re

MODE = "folder"

DATASET_PATH = r"..\train"

RE_LABEL = re.compile("[\\w]+_([\\d])+_([\\d])+")

IMG_HEIGHT = 480
IMG_WIDTH = 640
CHANNELS = 3

__IMAGE = None

def read_images(dataset_path, mode, batch_size, head = "rgb"):
	imagepaths, labels = list(), list()
	if mode == "file":
		return None
	elif mode == "folder":
		samples = os.walk(dataset_path).__next__()[2]
		for sample in samples:
			if '.png' in sample:
				if head in sample:
					imagepaths.append(os.path.join(dataset_path,sample))
					labels.append([int(i) for i in RE_LABEL.findall(sample)[0]])
		#print(imagepaths)
	else:
		raise Exception("Unknown mode.")

	imagepaths = tf.convert_to_tensor(imagepaths,dtype=tf.string)
	labels = tf.convert_to_tensor(labels, dtype=tf.uint16)

	image,label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)


	image = tf.read_file(image)
	image = tf.image.decode_png(image,channels=CHANNELS)
	global __IMAGE
	__IMAGE = image

	image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

	image = image * 1.0/127.5 - 1.0

	X, Y = tf.train.batch([image, label], batch_size=batch_size,
		capacity=batch_size * 8, num_threads=4)
	return X,Y
		



if __name__ == "__main__":
	read_images(DATASET_PATH,MODE,64)
	import matplotlib.pyplot as plt
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		image = sess.run(__IMAGE)
		print(image.shape)
		plt.imshow(image)
		plt.show()
