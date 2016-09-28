'''
Created on Jul 5, 2016

@author: kekim
'''

from __future__ import absolute_import
from __future__ import division

import numpy as np

import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras.layers import Embedding, Layer
from keras import backend as K


class ZonzEmbedding(Embedding):
	def call(self, x, mask=None):
		if 0. < self.dropout < 1.:
			retain_p = 1. - self.dropout
			B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
			B = K.expand_dims(B)
			W = K.in_train_phase(self.W * B, self.W)
		else:
			W = self.W
		M = K.concatenate([K.zeros((1,)), K.ones((self.input_dim - 1,))], axis=0)
		M = K.expand_dims(M)
		out = K.gather(W * M, x)
		return out


class MaskEatingLambda(Layer):
	'''Used for evaluating an arbitrary Theano / TensorFlow expression
	on the output of the previous layer.
	# Examples
	```python
		# add a x -> x^2 layer
		model.add(Lambda(lambda x: x ** 2))
	```
	```python
		# add a layer that returns the concatenation
		# of the positive part of the input and
		# the opposite of the negative part
		def antirectifier(x):
			x -= K.mean(x, axis=1, keepdims=True)
			x = K.l2_normalize(x, axis=1)
			pos = K.relu(x)
			neg = K.relu(-x)
			return K.concatenate([pos, neg], axis=1)
		def antirectifier_output_shape(input_shape):
			shape = list(input_shape)
			assert len(shape) == 2  # only valid for 2D tensors
			shape[-1] *= 2
			return tuple(shape)
		model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
	```
	# Arguments
		function: The function to be evaluated.
			Takes one argument: the output of previous layer
		output_shape: Expected output shape from function.
			Could be a tuple or a function of the shape of the input
		arguments: optional dictionary of keyword arguments to be passed
			to the function.
	# Input shape
		Arbitrary. Use the keyword argument input_shape
		(tuple of integers, does not include the samples axis)
		when using this layer as the first layer in a model.
	# Output shape
		Specified by `output_shape` argument.
	'''

	def __init__(self, function, output_shape=None, arguments={}, **kwargs):
		self.function = function
		self.arguments = arguments
		self.supports_masking = True
		if output_shape is None:
			self._output_shape = None
		elif type(output_shape) in {tuple, list}:
			self._output_shape = tuple(output_shape)
		else:
			if not hasattr(output_shape, '__call__'):
				raise Exception('In Lambda, `output_shape` '
								'must be a list, a tuple, or a function.')
			self._output_shape = output_shape
		super(MaskEatingLambda, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		if self._output_shape is None:
			# if TensorFlow, we can infer the output shape directly:
			if K._BACKEND == 'tensorflow':
				if type(input_shape) is list:
					xs = [K.placeholder(shape=shape) for shape in input_shape]
					x = self.call(xs)
				else:
					x = K.placeholder(shape=input_shape)
					x = self.call(x)
				if type(x) is list:
					return [K.int_shape(x_elem) for x_elem in x]
				else:
					return K.int_shape(x)
			# otherwise, we default to the input shape
			return input_shape
		elif type(self._output_shape) in {tuple, list}:
			nb_samples = input_shape[0] if input_shape else None
			return (nb_samples,) + tuple(self._output_shape)
		else:
			shape = self._output_shape(input_shape)
			if type(shape) not in {list, tuple}:
				raise Exception('output_shape function must return a tuple')
			return tuple(shape)

	def call(self, x, mask=None):
		arguments = self.arguments
		arg_spec = inspect.getargspec(self.function)
		if 'mask' in arg_spec.args:
			arguments['mask'] = mask
		return self.function(x, **arguments)

	def get_config(self):
		py3 = sys.version_info[0] == 3

		if isinstance(self.function, python_types.LambdaType):
			if py3:
				function = marshal.dumps(self.function.__code__).decode('raw_unicode_escape')
			else:
				function = marshal.dumps(self.function.func_code).decode('raw_unicode_escape')
			function_type = 'lambda'
		else:
			function = self.function.__name__
			function_type = 'function'

		if isinstance(self._output_shape, python_types.LambdaType):
			if py3:
				output_shape = marshal.dumps(self._output_shape.__code__)
			else:
				output_shape = marshal.dumps(self._output_shape.func_code)
			output_shape_type = 'lambda'
		elif callable(self._output_shape):
			output_shape = self._output_shape.__name__
			output_shape_type = 'function'
		else:
			output_shape = self._output_shape
			output_shape_type = 'raw'

		config = {'function': function,
				  'function_type': function_type,
				  'output_shape': output_shape,
				  'output_shape_type': output_shape_type,
				  'arguments': self.arguments}
		base_config = super(MaskEatingLambda, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_mask(self, x, input_mask=None):
		return None

	@classmethod
	def from_config(cls, config):
		function_type = config.pop('function_type')
		if function_type == 'function':
			function = globals()[config['function']]
		elif function_type == 'lambda':
			function = marshal.loads(config['function'].encode('raw_unicode_escape'))
			function = python_types.FunctionType(function, globals())
		else:
			raise Exception('Unknown function type: ' + function_type)

		output_shape_type = config.pop('output_shape_type')
		if output_shape_type == 'function':
			output_shape = globals()[config['output_shape']]
		elif output_shape_type == 'lambda':
			output_shape = marshal.loads(config['output_shape'])
			output_shape = python_types.FunctionType(output_shape, globals())
		else:
			output_shape = config['output_shape']

		config['function'] = function
		config['output_shape'] = output_shape
		return cls(**config)


# to be used in the above lambda layer
def lambda_mask_sum(x, mask=None):
	return K.batch_dot(x, mask, axes=1)


def lambda_mask_mean(x, mask=None):
	return K.batch_dot(x, mask, axes=1) / K.sum(x, axis=1)


def lambda_mask_reverse(x, mask=None):
	return x[:, ::-1, :] * K.expand_dims(mask[::-1])


def lambda_mask_zero(x, mask=None):
	return x * K.expand_dims(mask)


# [attention, input]
def lambda_attended(x, mask=None):
	x0 = K.expand_dims(x[0], dim=-1)
	x1 = K.expand_dims(x[1], dim=2)
	return x0 * x1


def time_softmax(x):
	ndim = K.ndim(x)
	if ndim == 3:
		e = K.exp(x - K.max(x, axis=1, keepdims=True))
		s = K.sum(e, axis=1, keepdims=True)
		return e / s
	else:
		raise Exception('Cannot apply time_softmax to a tensor that is no 3D. '+
						'Here, ndim= '+str(ndim))

