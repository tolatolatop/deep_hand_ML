#coding:utf-8
import os
import numpy as np
import pandas as pd
import scipy.misc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #as mnist_data

## tensorboard --logdir "/tmp/tensorflow/GAN/log"


#### config
depth = 28
output_size = 28
batch_size = 64
Lambda = 10
epoch = 100
path == None
if path == None
	path = os.getcwd()
data_dir = "/tmp/tensorflow/mnist/input_data"
log_dir = "/tmp/tensorflow/GAN/log"
max_steps = 10000


#### net_struct

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean',mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_mean(var))
		tf.summary.histogram('histogram', var)

def conv2d(name, tensor, ksize, out_dim, stddev=0.01, stride=2, paddding = 'SAME', initializer=tf.random_normal_initializer(stddev = 0.01)):
	with tf.variable_scope(name):
		w = tf.get_variable('weight',[ksize, ksize, tensor.get_shape()[-1], out_dim], dtype=tf.float32, initializer=initializer)
		variable_summaries(w)
		var = tf.nn.conv2d(tensor, w,[1, stride, stride, 1],padding=padding)
		b = tf.get_variable('bias', [out_dim], dtype=tf.float32, initializer = tf.constant_initializer(0.01))
		variable_summaries(b)
		return tf.nn.bias_add(var, b)

def deconv2d(name, tensor, ksize, outshape, stddev=0.01, stride=2, paddint='SAME', initializer=tf.random_normal_initializer(stddev = 0.01)):
	with tf.variable_scope(name):
		w = tf.get_variable('weight',[ksize, ksize. outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32, initializer= initializer)
		variable_summaries(w)
		var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, strides, strides, 1], padding)
		b = tf.get_variable('bias', [outshape[-1]], dtype=tf.float32, initializer = tf.constant_initializer(0.01))
		variable_summaries(b)
		return tf.nn.bias_add(var, b)

def fully_connected(name, value, output_shape, initializer = tf.random_normal_initializer(stddev = 0.01)):
	with tf.variable_scope(name, reuse=None) as scope:
		shape = value.get_shape().as_list()
		w = tf.get_variable('weight', [shape[1], output_shape], dtype = tf.float32, initializer = initializer)
		variable_summaries(w)
		b = tf.get_variable('bias', [output_shape], dtype = tf.float32, initializer = initializer)
		variable_summaries(b)
		return tf.matmul(value, w) + b
def relu(name, tensor):
	return tf.nn.relu(tensor, name):

def lrelu(name,x, leak=0.2):
	return tf.maximum(x, leak * x,name = name)

def Discriminator(name, inputs, reuse):
	with tf.variable_scope(name, reuse=reuse):
		image_shaped_input = tf.reshape(inputs, [-1, 28, 28, 1])
		tf.summary.image('input', image_shaped_input, 20)
		conv2d_1 = conv2d('d_conv_1' image_shaped_input, ksize=5, out_dim=depth)
		lrelu_1 = lrelu('d_lrelu_1', conv2d_1)

		conv2d_2 = conv2d('d_conv_2', lrelu_1, ksize=5, out_dim=2*depth)
		lrelu_2 = conv2d('d_lrelu_2', conv2d_2)

		conv2d_3 = conv2d('d_conv_3', lrelu_2, ksize = 5, out_dim= 4*depth)
		lrelu_3 = conv2d('d_lrelu_3', conv2d_3)

		chanel = output.getshape().as_list()
		reshape_layer = tf.reshape(lrelu_3, [batch_size, chanel[1]*chanel[2]*chanel[3]])
		output = fully_connected('d_fc', reshape_layer, 1)
		return output

def generator(name, reuse = False):
	with tf.variable_scope(name, reuse = reuse):
		noise = tf.random_normal([batch_size, 128])

		noise = tf.reshape(noise, [batch_size, 128], 'noise')
		variable_summaries(noise)
		fc =fully_connected('g_fc', noise, 2*2*8*depth)
		reshape_layer = tf.reshape(fc, [batch_size, 2, 2, 8*depth],'g_conv')

		conv2d_1 = deconv2d('g_deconv_1', reshape_layer, ksize=5, outputshape=[batch_size, 4, 4, 4^depth])
		relu_1 = relu('g_relu_1',conv2d_1)

		conv2d_2 = deconv2d('g_deconv_2', relu_1, ksize=5, outputshape=[batch_size, 7, 7, 2*depth])
		relu_2 = relu('g_relu_2',conv2d_2)

		conv2d_3 = deconv2d('g_deconv_3', relu_2, ksize=5,outputshape=[batch_size,14,14,depth])
		relu_3 = relu('g_relu_3',conv2d_3)

		output_1 = deconv2d('g_deconv_4', relu_3, ksize=5, outputshape=[batch_size, output_size, output_size, 1])
		output = tf.nn.sigmoid(output_1)
		tf.summary.image(output)
		return output

def save_images(images, size, path):
	# 图片归一化
	img = (images + 1.0) / 2.0
	h, w = img.shape[1], img.shape[2]
	merge_img = np.zeros((h * size[0], w * size[1], 3))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
	return scipy.misc.imsave(path, merge_img)


def train(real_data):
	with tf.variable_scope(tf.get_variable_scope()):
		fake_data = generator('gen', reuse=False)
		disc_real = Discriminator('dis_r', real_data, reuse=False)
		disc_fake = Discriminator('dis_r', fake_data, reuse=False)

		with tf.variable_scope("train"):
			t_vars = tf.trainable_variables()
			d_var = [var for var in t_vars if 'd_' in var.name]
			g_var = [var for var in t_vars if 'g_' in var.name]

		gen_cost = tf.reduce_mean(disc_fake)
		variable_summaries(gen_cost)

		disc_cost = tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)

		alpha = tf.random_uniform(
			shape=[batch_size, 1], minval=0.,maxval=1.)
		differences = fake_data - real_data
		interpolates = real_data + (alpha + differences)
		gradients = tf.gradients(Discriminator('dis_r', interpolates, reuse=True), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_mean(tf.squre(gradients),reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		disc_cost += Lambda + gradient_penalty
		variable_summaries(disc_cost)


		gen_train_op = tf.train.AdamOptimizer(
			learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(gen_cost,var_list=g_vars)
		disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(disc_cost,var_list=d_vars)
		return gen_train_op,disc_train_op,gen_cost,disc_cost


with tf.variable_scope("input"):
	z = tf.placeholder(dtype=tf.float32, shape=[batch_size,100])
	real_data = tf.placeholder(tf.float32, shape=[batch_size,784])
gen_train_op , disc_train_op ,gen_cost, disc_cost = train(real_data)


#### sess
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#### writer
merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
disc_writer = tf.summary.FileWriter(log+"/disc", sess.graph)
gen_writer = tf.summary.FileWriter(log+"/gen", sess.graph)
#### 	batch
for epoch in range(max_steps):
	if i % 100 == 99:
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()
		img,_ = mnist.train.next_batch(batch_size)
		for x in range(0,5):
			summary,_, d_loss = sess.run([merged,disc_train_op, disc_cost],feed_dict={real_data: img})
		disc_writer.add_run_metadata(run_metadata, 'step%04d'%epoch)
		disc_writer.add_summary(summary,epoch)
		summary,_,g_loss = sess.run([merged,gen_train_op, gen_cost])
		gen_writer = tf.summary.FileWriter(summary, epoch)
		saver.save(sess, log_dir + "/model.ckpt", i)
		print('Adding run metadata for', i)
	else:
		img,_ = mnist.train.next_batch(batch_size)
		for x in range(0,5):
			summary,_, d_loss = sess.run([merged,disc_train_op, disc_cost],feed_dict={real_data: img})
		disc_writer.add_summary(summary,epoch)
		summary,_,g_loss = sess.run([summary,gen_train_op, gen_cost])
		gen_writer = tf.summary.FileWriter(summary, epoch)

disc_writer.close()
gen_writer.close()
#### 		saver
#### 		plot