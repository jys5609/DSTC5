import re
import gensim
import theano
from theano import tensor as T
import scripts as sc
import numpy as np
import random
import pickle
from collections import defaultdict

from keras.preprocessing import sequence as ksq
from keras.models import Sequential, Model
from keras.layers import *
from keras.constraints import *
from utils import *
import keras.utils.np_utils as np_utils
import keras.optimizers
from keras.callbacks import ModelCheckpoint

####### configuration
np.random.seed(1)
#######

def ll():
	return defaultdict(list)
def ls():
	return defaultdict(set)

def cosine_sim(y_pred, y_true):
	pred_norm = T.sum(T.square(y_pred), axis=2, keepdims=True)
	pred_norm = T.switch(pred_norm <= 0, 1.0, pred_norm)
	y_pred = y_pred / T.sqrt(pred_norm)
	true_norm = T.sum(T.square(y_true), axis=2, keepdims=True)
	true_norm = T.switch(true_norm <= 0, 1.0, true_norm)
	y_true = y_true / T.sqrt(true_norm)
	return -K.mean(K.sum((y_true * y_pred), axis=2))


class Tracker():
	def __init__(self, dstctype, embed_dim, segment_length):
		self.exist_model = False
		general_filename = 'dstc' + str(dstctype) + '_general.pkl'

		pkl_file = open(general_filename, 'rb')

		self.slots = pickle.load(pkl_file)
		self.slot_loc = pickle.load(pkl_file)
		self.slot_value = pickle.load(pkl_file)
		self.slot_value_vector = pickle.load(pkl_file)

		pkl_file.close()

		self.slot_num = len(self.slots)

		self.embed_dim = embed_dim
		self.segment_length = segment_length

		self.info_loc = (np.array(self.slots)[:, 1] == 'INFO').astype(int)
		self.info_loc = self.info_loc.reshape((self.slot_num, 1))
		self.info_loc = np.tile(self.info_loc, self.embed_dim)

		self.exist_model = False

	def load_data(self, dstctype, datatype, uttr_accumulate):

		if uttr_accumulate:
			filename = 'dstc' + str(dstctype) + '_' + datatype + '_acc.pkl'
		else:
			filename = 'dstc' + str(dstctype) + '_' + datatype + '.pkl'

		pkl_file = open(filename, 'rb')

		SUID = pickle.load(pkl_file)
		X = pickle.load(pkl_file)
		X_ins = pickle.load(pkl_file)
		X_text = pickle.load(pkl_file)
		X_topic = pickle.load(pkl_file)
		Y = pickle.load(pkl_file)
		Y_text = pickle.load(pkl_file)
		pkl_file.close()

		X = ksq.pad_sequences(X, maxlen=self.segment_length, dtype='float32')
		X_ins = ksq.pad_sequences(X_ins, maxlen=self.segment_length, dtype='float32')
		Y = np.array(Y)

		return (SUID, X, X_ins, X_text, X_topic, Y, Y_text)


	def lambda_tile_embed(self, x):
		return T.tile(x, self.embed_dim)

	def lambda_tile_embed_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(input_shape) == 3
		shape[-1] *= self.embed_dim
		return tuple(shape)

	def lambda_sum(self, x):
		return K.sum(x, axis=1, keepdims=True)

	def lambda_sum_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(input_shape) == 3
		return tuple([shape[0], 1, shape[2]])

	def tile_slot(self, x):
		return T.repeat(x, self.slot_num, axis=1)

	def tile_slot_output_shape(self, input_shape):
		shape = list(input_shape)
		shape[1] *= self.slot_num
		return tuple(shape)

	def create_model(self, learning_rate, lstm_units, dropout1, dropout2):

		input_x = Input(shape=(self.segment_length, self.embed_dim,), dtype='float32')
		input_ins = Input(shape=(self.segment_length, self.slot_num))
		input_info_msk = Input(shape=(self.slot_num, self.embed_dim,), dtype='float32')  # (None, 30, 100)

		x = merge([input_x, input_ins], mode='concat', concat_axis=-1)
		x = Masking()(x)
		x = Dropout(dropout1)(x)
		xf = LSTM(lstm_units, return_sequences=True)(x)
		xf = MaskEatingLambda(lambda_mask_zero)(xf)
		xb = LSTM(lstm_units, return_sequences=True, go_backwards=True)(x)
		xb = MaskEatingLambda(lambda_mask_reverse)(xb)
		x = merge([xf, xb], mode='concat', concat_axis=-1)
		x = Masking()(x)
		x = Dropout(dropout2)(x)

		x1_info = TimeDistributed(Dense(self.embed_dim))(x)
		x1_info = MaskEatingLambda(lambda_mask_zero)(x1_info)  # (None, 240, 100)
		x1_info = Lambda(self.lambda_sum, output_shape=self.lambda_sum_output_shape)(x1_info)
		x1_info = Reshape((1, self.embed_dim,))(x1_info)  # (None, 1, 100)
		x1_info = Lambda(self.tile_slot, output_shape=self.tile_slot_output_shape)(x1_info)  # (None, 30, 100)
		x1_info = merge([x1_info, input_info_msk], mode='mul')  # (None, 30, 100)

		x1_each = [TimeDistributed(Dense(1))(x) for i in range(self.slot_num)]  # (None, 240, 1)
		x1_each = [MaskEatingLambda(lambda_mask_zero)(x1_each[i]) for i in range(self.slot_num)]
		x1_each = [Reshape((self.segment_length,))(x1_each[i]) for i in range(self.slot_num)]  # (None, 240)
		x1_each = [Masking()(x1_each[i]) for i in range(self.slot_num)]
		x1_each = [Activation('softmax')(x1_each[i]) for i in range(self.slot_num)]  # (None, 240)
		x1_each = [MaskEatingLambda(lambda_mask_zero)(x1_each[i]) for i in range(self.slot_num)]
		x1_each = [Reshape((self.segment_length, 1,))(x1_each[i]) for i in
				   range(self.slot_num)]  # (None, 240, 1) * 30

		out_prob = merge(x1_each, mode='concat', concat_axis=-1, name='out_prob')

		x2_each = [Lambda(self.lambda_tile_embed, output_shape=self.lambda_tile_embed_output_shape)(x1_each[i]) for i in
				   range(self.slot_num)]
		x2_each = [merge([x2_each[i], input_x], mode='mul') for i in range(self.slot_num)]
		x2_each = [Lambda(self.lambda_sum, output_shape=self.lambda_sum_output_shape)(x2_each[i]) for i in
				   range(self.slot_num)]
		x2_each = [Reshape((1, self.embed_dim,))(x2_each[i]) for i in range(self.slot_num)]

		x2_each = merge(x2_each, mode='concat', concat_axis=1)  # (None, 30, 100)

		out_embed = merge([x2_each, x1_info], mode='sum', name='out_embed')  # (None, 30, 100)

		self.model = Model(input=[input_x, input_ins, input_info_msk], output=[out_embed, out_prob])
		optim = keras.optimizers.Adam(lr=learning_rate)
		self.model.compile(loss={'out_embed': cosine_sim, 'out_prob': 'mse'}, optimizer=optim, metrics=[],
						   loss_weights={'out_embed': 1.0, 'out_prob': 0.0})
		#self.model.summary()
		self.exist_model = True


		'''
		dense_num = 5

		input_x = Input(shape=(self.segment_length, self.embed_dim,), dtype='float32')	# (None, segment_length, embed_dim)
		input_ins = Input(shape=(self.segment_length, self.slots_num)) # dtype='int'
		input_info_msk = Input(shape=(self.slots_num, self.embed_dim,), dtype='float32')

		x = merge([input_x, input_ins], mode='concat', concat_axis=-1)
		x = Masking()(x)
		x = Dropout(dropout1)(x)
		xf = LSTM(lstm_units, return_sequences=True)(x)
		xf = MaskEatingLambda(lambda_mask_zero)(xf)
		xb = LSTM(lstm_units, return_sequences=True, go_backwards=True)(x)
		xb = MaskEatingLambda(lambda_mask_reverse)(xb)
		x = merge([xf, xb], mode='concat', concat_axis=-1)
		x = Masking()(x) # (None, segment_length, 2*lstm_units)
		x = Dropout(dropout2)(x)

		x1_info = TimeDistributed(Dense(self.embed_dim))(x) # (None, segment_length, embed_dim) # TODO: multiple layers?
		x1_info = MaskEatingLambda(lambda_mask_zero)(x1_info) # (None, segment_length, embed_dim)
		x1_info = Lambda(self.lambda_sum, output_shape=self.lambda_sum_output_shape)(x1_info) # (None, 1, embed_dim)

		#x1_info = Reshape((1, self.embed_dim,))(x1_info)


		x1_each = TimeDistributed(Dense(self.slots_num))(x) # (None, segment_length, slots_num)
		#x1_each = TimeDistributed(Dense(self.slots_num))(x1_each) # TODO: multiple dense layers?
		out_prob = Activation(time_softmax)(x1_each)	# (None, segment_length, slots_num)




		optim = keras.optimizers.Adam(lr=learning_rate)

		self.model = Model(input=[input_x, input_ins, input_info_msk], output=x1_info)
		self.model.compile(loss='mse', optimizer=optim)
		#self.model.compile(loss={}, optimizer=optim, metrics=[])
		self.model.summary()
		self.exist_model = True
		'''

	def train(self, X, X_ins, Y, valid_X, valid_X_ins, valid_Y, nb_epoch, weight_file):
		assert self.exist_model

		train_info_loc = np.array([self.info_loc for i in range(X.shape[0])])
		valid_info_loc = np.array([self.info_loc for i in range(valid_X.shape[0])])

		train_temp_prob = np.zeros((X.shape[0], self.segment_length, self.slot_num))
		valid_temp_prob = np.zeros((valid_X.shape[0], self.segment_length, self.slot_num))

		checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True)
		self.model.fit([X, X_ins, train_info_loc], [Y, train_temp_prob], batch_size=256, nb_epoch=nb_epoch, validation_split=0., validation_data=([valid_X, valid_X_ins, valid_info_loc], [valid_Y, valid_temp_prob]), verbose=1, callbacks=[checkpointer])
		self.model.save_weights(weight_file, overwrite=True)
		print 'weight saved to '+weight_file
