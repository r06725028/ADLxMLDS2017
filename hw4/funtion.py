import math
import numpy as np 
import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.python.framework import ops
from scipy import misc
from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize
import random
import skimage
import skimage.io
import skimage.transform
import os
import theano
import theano.tensor as tensor
import pickle as pkl
import copy
import nltk


profile = False

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias
def binary_cross_entropy(preds, targets, name=None):
	eps = 1e-12
	with ops.op_scope([preds, targets], name, "bce_loss") as name:
		preds = ops.convert_to_tensor(preds, name="preds")
		targets = ops.convert_to_tensor(targets, name="targets")
		return tf.reduce_mean(-(targets * tf.log(preds + eps) +(1. - targets) * tf.log(1. - preds + eps)))
def conv_cond_concat(x, y):
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])
def conv2d(input_, output_dim,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		return conv
def deconv2d(input_, output_shape,
			 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
			 name="deconv2d", with_w=False):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
							initializer=tf.random_normal_initializer(stddev=stddev))	
		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])
		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
		if with_w:
			return deconv, w, biases
		else:
			return deconv
def minibatch(input_layer, num_kernels, kernel_dim): 
	batch_size = input_layer.shape[0]
	num_features = input_layer.shape[1]

	W = tf.get_variable("minibatch_w", [num_features, num_kernels * kernel_dim],\
		tf.float32, tf.random_normal_initializer(stddev=0.02))
	b = tf.get_variable("minibatch_b", [num_kernels], initializer=tf.constant_initializer(0.0))

	activation = tf.matmul(input_layer, W)
	activation = tf.reshape(activation, (-1, num_kernels, kernel_dim))
	diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
	abs_diffs = tf.reduce_sum(tf.abs(diffs), reduction_indices=[2])
	minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), reduction_indices=[2])
	return minibatch_features + b
#############################################################################################################
class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon = epsilon
			self.momentum = momentum
			self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
			self.name = name

	def __call__(self, x, train=True):
		shape = x.get_shape().as_list()
		if train:
			with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
				self.beta = tf.get_variable("beta", [shape[-1]],initializer=tf.constant_initializer(0.))
				self.gamma = tf.get_variable("gamma", [shape[-1]],initializer=tf.random_normal_initializer(1., 0.02))
				try:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
				except:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')	
				ema_apply_op = self.ema.apply([batch_mean, batch_var])
				self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
				with tf.control_dependencies([ema_apply_op]):
					mean, var = tf.identity(batch_mean), tf.identity(batch_var)
		else:
			mean, var = self.ema_mean, self.ema_var
		normed = tf.nn.batch_norm_with_global_normalization(
				x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
		return normed

#########################################################################################################
class wgan:
	def __init__(self, options):
		self.options = options
		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')
		self.g_bn3 = batch_norm(name='g_bn3')
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')
		self.d_bn4 = batch_norm(name='d_bn4')

	def build_model(self):
		img_size = self.options['image_size']
		t_real_image = tf.placeholder('float32', [self.options['batch_size'], img_size, img_size, 3 ], name = 'real_image')
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'], img_size, img_size, 3 ], name = 'wrong_image')
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])

		fake_image = self.generator(t_z, t_real_caption)

		disc_real_image, disc_real_image_logits = self.discriminator(t_real_image, t_real_caption)
		disc_wrong_image, disc_wrong_image_logits = self.discriminator(t_wrong_image, t_real_caption, reuse=True)
		disc_fake_image, disc_fake_image_logits = self.discriminator(fake_image, t_real_caption, reuse=True)

		g_loss = 0.5 * tf.reduce_mean((disc_fake_image_logits - tf.ones_like(disc_fake_image)) ** 2)
		d_loss1 = 0.5 * tf.reduce_mean((disc_real_image_logits - tf.ones_like(disc_real_image)) ** 2)
		d_loss2 = 0.5 * tf.reduce_mean((disc_wrong_image_logits - tf.zeros_like(disc_wrong_image)) ** 2)
		d_loss3 = 0.5 * tf.reduce_mean((disc_fake_image_logits - tf.zeros_like(disc_fake_image)) ** 2)
		d_loss = d_loss1 + d_loss2 + d_loss3
	
		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		regularizers = 0
		for d_weight in d_vars:
			regularizers += tf.nn.l2_loss(d_weight)
		d_loss = tf.reduce_mean(d_loss + 0.01 * regularizers)

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss}

		outputs = {
			'generator' : fake_image}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'disc_real_image_logits' : disc_real_image_logits,
			'disc_wrong_image_logits' : disc_wrong_image,
			'disc_fake_image_logits' : disc_fake_image_logits}

		return input_tensors, variables, loss, outputs, checks

	def build_generator(self):
		img_size = self.options['image_size']
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		fake_image = self.sampler(t_z, t_real_caption)

		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z}

		outputs = {
			'generator' : fake_image}

		return input_tensors, outputs

	def sampler(self, t_z, t_text_embedding):
		tf.get_variable_scope().reuse_variables()

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

		reduced_text_embedding = lrelu( linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat([t_z, reduced_text_embedding], axis=1)
		z_ = linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0, train = False))

		h1 = deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1, train = False))

		h2 = deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2, train = False))

		h3 = deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3, train = False))

		h4 = deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

		return (tf.tanh(h4)/2. + 0.5)

	def generator(self, t_z, t_text_embedding):

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

		reduced_text_embedding = lrelu( linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat([t_z, reduced_text_embedding], axis=1)
		z_ = linear(z_concat, self.options['gf_dim'] * 8 * s16 * s16, 'g_h0_lin')

		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0))
		h1 = deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim'] * 4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1))
		h2 = deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim'] * 2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2))
		h3 = deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim'] * 1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3))
		h4 = deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		return (tf.tanh(h4) / 2. + 0.5)

	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			with tf.variable_scope(tf.get_variable_scope()) as scope:
				scope.reuse_variables()
				h0 = lrelu(conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) 
				h1 = lrelu(self.d_bn1(conv2d(h0, self.options['df_dim'] * 2, name = 'd_h1_conv'))) 
				h2 = lrelu(self.d_bn2(conv2d(h1, self.options['df_dim'] * 4, name = 'd_h2_conv'))) 
				h3 = lrelu(self.d_bn3(conv2d(h2, self.options['df_dim'] * 8, name = 'd_h3_conv'))) 

				reduced_text_embeddings = lrelu(linear(t_text_embedding, self.options['t_dim'], 'd_embedding')) 

				reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1) 

				reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2) 

				tiled_embeddings = tf.tile(reduced_text_embeddings, [1, 4, 4, 1], name='tiled_embeddings') 
				h3_concat = tf.concat( [h3, tiled_embeddings], axis=3, name='h3_concat')
				h3_new = lrelu( self.d_bn4(conv2d(h3_concat, self.options['df_dim'] * 8, 1, 1, 1, 1, name='d_h3_conv_new'))) 	
				h4 = tf.reshape(h3_new, [self.options['batch_size'], -1]) 
				h4 = linear(minibatch(h4, num_kernels=150, kernel_dim=8), 1, 'd_h3_lin')
		
				return tf.nn.sigmoid(h4), h4
		else:
			h0 = lrelu(conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) 
			h1 = lrelu(self.d_bn1(conv2d(h0, self.options['df_dim'] * 2, name = 'd_h1_conv'))) 
			h2 = lrelu(self.d_bn2(conv2d(h1, self.options['df_dim'] * 4, name = 'd_h2_conv'))) 
			h3 = lrelu(self.d_bn3(conv2d(h2, self.options['df_dim'] * 8, name = 'd_h3_conv'))) 

			reduced_text_embeddings = lrelu(linear(t_text_embedding, self.options['t_dim'], 'd_embedding')) 
			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1) 
			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)

			tiled_embeddings = tf.tile(reduced_text_embeddings, [1, 4, 4, 1], name='tiled_embeddings')
			h3_concat = tf.concat( [h3, tiled_embeddings], axis=3, name='h3_concat')
			h3_new = lrelu( self.d_bn4(conv2d(h3_concat, self.options['df_dim'] * 8, 1, 1, 1, 1, name='d_h3_conv_new'))) 
			h4 = tf.reshape(h3_new, [self.options['batch_size'], -1]) 
			h4 = linear(minibatch(h4, num_kernels=150, kernel_dim=8), 1, 'd_h3_lin') 

			return tf.nn.sigmoid(h4), h4
##################################################################################################
profile = False

#-----------------------------------------------------------------------------#
# Specify model and table locations here
#-----------------------------------------------------------------------------#
path_to_models = 'skipthoughts/'
path_to_tables = 'skipthoughts/'
#-----------------------------------------------------------------------------#

path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'


def load_model():
	"""
	Load the model with saved tables
	"""
	# Load model options
	print('load pkl...')
	with open('%s.pkl'%path_to_umodel, 'rb') as f:
		uoptions = pkl.load(f)
	with open('%s.pkl'%path_to_bmodel, 'rb') as f:
		boptions = pkl.load(f)

	# Load parameters
	uparams = init_params(uoptions)
	uparams = load_params(path_to_umodel, uparams)
	utparams = init_tparams(uparams)
	bparams = init_params_bi(boptions)
	bparams = load_params(path_to_bmodel, bparams)
	btparams = init_tparams(bparams)

	# Extractor functions
	print('build encoder...')
	embedding, x_mask, ctxw2v = build_encoder(utparams, uoptions)
	f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')
	embedding, x_mask, ctxw2v = build_encoder_bi(btparams, boptions)
	f_w2v2 = theano.function([embedding, x_mask], ctxw2v, name='f_w2v2')

	# Tables
	#print('Loading tables...')
	utable, btable = load_tables()

	# Store everything we need in a dictionary
	#print('Packing up...')
	model = {}
	model['uoptions'] = uoptions
	model['boptions'] = boptions
	model['utable'] = utable
	model['btable'] = btable
	model['f_w2v'] = f_w2v
	model['f_w2v2'] = f_w2v2

	return model


def load_tables():
	"""
	Load the tables
	"""
	words = []
	utable = np.load(path_to_tables + 'utable.npy', encoding="latin1")
	btable = np.load(path_to_tables + 'btable.npy', encoding="latin1")
	f = open(path_to_tables + 'dictionary.txt', 'rb')
	for line in f:
		words.append(line.decode('utf-8').strip())
	f.close()
	utable = OrderedDict(zip(words, utable))
	btable = OrderedDict(zip(words, btable))
	return utable, btable


def encode(model, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):
	"""
	Encode sentences in the list X. Each entry will return a vector
	"""
	# first, do preprocessing
	X = preprocess(X)

	# word dictionary and init
	d = defaultdict(lambda : 0)
	for w in model['utable'].keys():
		d[w] = 1
	ufeatures = np.zeros((len(X), model['uoptions']['dim']), dtype='float32')
	bfeatures = np.zeros((len(X), 2 * model['boptions']['dim']), dtype='float32')

	# length dictionary
	ds = defaultdict(list)
	captions = [s.split() for s in X]
	for i,s in enumerate(captions):
		ds[len(s)].append(i)

	# Get features. This encodes by length, in order to avoid wasting computation
	for k in ds.keys():
		#if verbose:
			#print(k)
		numbatches = int(len(ds[k]) / batch_size + 1)
		for minibatch in range(numbatches):
			caps = ds[k][int(minibatch)::numbatches]

			if use_eos:
				uembedding = np.zeros((k+1, len(caps), model['uoptions']['dim_word']), dtype='float32')
				bembedding = np.zeros((k+1, len(caps), model['boptions']['dim_word']), dtype='float32')
			else:
				uembedding = np.zeros((k, len(caps), model['uoptions']['dim_word']), dtype='float32')
				bembedding = np.zeros((k, len(caps), model['boptions']['dim_word']), dtype='float32')
			for ind, c in enumerate(caps):
				caption = captions[c]
				for j in range(len(caption)):
					if d[caption[j]] > 0:
						uembedding[j,ind] = model['utable'][caption[j]]
						bembedding[j,ind] = model['btable'][caption[j]]
					else:
						uembedding[j,ind] = model['utable']['UNK']
						bembedding[j,ind] = model['btable']['UNK']
				if use_eos:
					uembedding[-1,ind] = model['utable']['<eos>']
					bembedding[-1,ind] = model['btable']['<eos>']
			if use_eos:
				uff = model['f_w2v'](uembedding, np.ones((len(caption)+1,len(caps)), dtype='float32'))
				bff = model['f_w2v2'](bembedding, np.ones((len(caption)+1,len(caps)), dtype='float32'))
			else:
				uff = model['f_w2v'](uembedding, np.ones((len(caption),len(caps)), dtype='float32'))
				bff = model['f_w2v2'](bembedding, np.ones((len(caption),len(caps)), dtype='float32'))
			if use_norm:
				for j in range(len(uff)):
					uff[j] /= norm(uff[j])
					bff[j] /= norm(bff[j])
			for ind, c in enumerate(caps):
				ufeatures[c] = uff[ind]
				bfeatures[c] = bff[ind]
	
	features = np.c_[ufeatures, bfeatures]
	return features


def preprocess(text):
	"""
	Preprocess text for encoder
	"""
	X = []
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	for t in text:
		sents = sent_detector.tokenize(t)
		result = ''
		for s in sents:
			tokens = word_tokenize(s)
			result += ' ' + ' '.join(tokens)
		X.append(result)
	return X


def nn(model, text, vectors, query, k=5):
	"""
	Return the nearest neighbour sentences to query
	text: list of sentences
	vectors: the corresponding representations for text
	query: a string to search
	"""
	qf = encode(model, [query])
	qf /= norm(qf)
	scores = np.dot(qf, vectors.T).flatten()
	sorted_args = np.argsort(scores)[::-1]
	sentences = [text[a] for a in sorted_args[:k]]
	print('QUERY: ' + query)
	print('NEAREST: ')
	for i, s in enumerate(sentences):
		print(s, sorted_args[i])


def word_features(table):
	"""
	Extract word features into a normalized matrix
	"""
	features = np.zeros((len(table), 620), dtype='float32')
	keys = table.keys()
	for i in range(len(table)):
		f = table[keys[i]]
		features[i] = f / norm(f)
	return features


def nn_words(table, wordvecs, query, k=10):
	"""
	Get the nearest neighbour words
	"""
	keys = table.keys()
	qf = table[query]
	scores = np.dot(qf, wordvecs.T).flatten()
	sorted_args = np.argsort(scores)[::-1]
	words = [keys[a] for a in sorted_args[:k]]
	print('QUERY: ' + query)
	print('NEAREST: ')
	for i, w in enumerate(words):
		print(w)


def _p(pp, name):
	"""
	make prefix-appended name
	"""
	return '%s_%s'%(pp, name)


def init_tparams(params):
	"""
	initialize Theano shared variables according to the initial parameters
	"""
	tparams = OrderedDict()
	for kk, pp in params.items():
		tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams


def load_params(path, params):
	"""
	load parameters
	"""
	pp = np.load(path)
	for kk, vv in params.items():
		if kk not in pp:
			warnings.warn('%s is not in the archive'%kk)
			continue
		params[kk] = pp[kk]
	return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'gru': ('param_init_gru', 'gru_layer')}

def get_layer(name):
	fns = layers[name]
	return (eval(fns[0]), eval(fns[1]))


def init_params(options):
	"""
	initialize all parameters needed for the encoder
	"""
	params = OrderedDict()

	# embedding
	params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

	# encoder: GRU
	params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
											  nin=options['dim_word'], dim=options['dim'])
	return params


def init_params_bi(options):
	"""
	initialize all paramters needed for bidirectional encoder
	"""
	params = OrderedDict()

	# embedding
	params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

	# encoder: GRU
	params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
											  nin=options['dim_word'], dim=options['dim'])
	params = get_layer(options['encoder'])[0](options, params, prefix='encoder_r',
											  nin=options['dim_word'], dim=options['dim'])
	return params


def build_encoder(tparams, options):
	"""
	build an encoder, given pre-computed word embeddings
	"""
	# word embedding (source)
	embedding = tensor.tensor3('embedding', dtype='float32')
	x_mask = tensor.matrix('x_mask', dtype='float32')

	# encoder
	proj = get_layer(options['encoder'])[1](tparams, embedding, options,
											prefix='encoder',
											mask=x_mask)
	ctx = proj[0][-1]

	return embedding, x_mask, ctx


def build_encoder_bi(tparams, options):
	"""
	build bidirectional encoder, given pre-computed word embeddings
	"""
	# word embedding (source)
	embedding = tensor.tensor3('embedding', dtype='float32')
	embeddingr = embedding[::-1]
	x_mask = tensor.matrix('x_mask', dtype='float32')
	xr_mask = x_mask[::-1]

	# encoder
	proj = get_layer(options['encoder'])[1](tparams, embedding, options,
											prefix='encoder',
											mask=x_mask)
	projr = get_layer(options['encoder'])[1](tparams, embeddingr, options,
											 prefix='encoder_r',
											 mask=xr_mask)

	ctx = tensor.concatenate([proj[0][-1], projr[0][-1]], axis=1)

	return embedding, x_mask, ctx


# some utilities
def ortho_weight(ndim):
	W = np.random.randn(ndim, ndim)
	u, s, v = np.linalg.svd(W)
	return u.astype('float32')


def norm_weight(nin,nout=None, scale=0.1, ortho=True):
	if nout == None:
		nout = nin
	if nout == nin and ortho:
		W = ortho_weight(nin)
	else:
		W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
	return W.astype('float32')


def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
	"""
	parameter init for GRU
	"""
	if nin == None:
		nin = options['dim_proj']
	if dim == None:
		dim = options['dim_proj']
	W = np.concatenate([norm_weight(nin,dim),
						   norm_weight(nin,dim)], axis=1)
	params[_p(prefix,'W')] = W
	params[_p(prefix,'b')] = np.zeros((2 * dim,)).astype('float32')
	U = np.concatenate([ortho_weight(dim),
						   ortho_weight(dim)], axis=1)
	params[_p(prefix,'U')] = U

	Wx = norm_weight(nin, dim)
	params[_p(prefix,'Wx')] = Wx
	Ux = ortho_weight(dim)
	params[_p(prefix,'Ux')] = Ux
	params[_p(prefix,'bx')] = np.zeros((dim,)).astype('float32')

	return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):
	"""
	Forward pass through GRU layer
	"""
	nsteps = state_below.shape[0]
	if state_below.ndim == 3:
		n_samples = state_below.shape[1]
	else:
		n_samples = 1

	dim = tparams[_p(prefix,'Ux')].shape[1]

	if mask == None:
		mask = tensor.alloc(1., state_below.shape[0], 1)

	def _slice(_x, n, dim):
		if _x.ndim == 3:
			return _x[:, :, n*dim:(n+1)*dim]
		return _x[:, n*dim:(n+1)*dim]

	state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
	state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
	U = tparams[_p(prefix, 'U')]
	Ux = tparams[_p(prefix, 'Ux')]

	def _step_slice(m_, x_, xx_, h_, U, Ux):
		preact = tensor.dot(h_, U)
		preact += x_

		r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
		u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

		preactx = tensor.dot(h_, Ux)
		preactx = preactx * r
		preactx = preactx + xx_

		h = tensor.tanh(preactx)

		h = u * h_ + (1. - u) * h
		h = m_[:,None] * h + (1. - m_)[:,None] * h_

		return h

	seqs = [mask, state_below_, state_belowx]
	_step = _step_slice

	rval, updates = theano.scan(_step,
								sequences=seqs,
								outputs_info = [tensor.alloc(0., n_samples, dim)],
								non_sequences = [tparams[_p(prefix, 'U')],
												 tparams[_p(prefix, 'Ux')]],
								name=_p(prefix, '_layers'),
								n_steps=nsteps,
								profile=profile,
								strict=True)
	rval = [rval]
	return rval
###########################################################################################
def load_image_array(image_file, image_size):
	img = skimage.io.imread(image_file)
	
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'uint8')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = skimage.transform.resize(img, (image_size, image_size))

	if random.random() > 0.5:
		img_resized = np.fliplr(img_resized)
	
	
	return img_resized.astype('float32')

