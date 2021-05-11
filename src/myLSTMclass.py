#!/usr/bin/env python
#! author: GC @ 11/25/2020 customized class for stacked RNN layers

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed

def RNN_VE(strain_arr, RNNmodel, states=None):
    """ RNN model of VE constitutive law of one sequence
    Inputs: strains with the shape (timesteps, n_features) 2-dim inputs
    Outputs: the stress prediction values
    """
    stress = []

    # loop around the timesteps
    for i in range(strain_arr.shape[0]):
        inputs = strain_arr[i,:]
        inputs=np.reshape(inputs, (1, 1, inputs.shape[0])) # inputs should be in 3-dim

        outputs, h_state0, c_state0, h_state1, c_state1 = RNNmodel.pred(inputs, states)
        istress = K.get_value(outputs)
        stress.append(istress)
        states = [h_state0, c_state0, h_state1, c_state1]

    return np.asarray(stress)

class myLSTM():
	def __init__(self, n_hidden, n_features, n_out, model_file):

		self._n_hidden = n_hidden
		self._n_features = n_features
		self._n_out = n_out
		self._model_file = model_file
		# self._initial_states = [tf.zeros((1, n_hidden)) for i in range(4)]

		# obtain the weights
		best_model = load_model(self._model_file)
		weights_list = best_model.weights
		# for the LSTM cell
		self._kernel_0 = K.get_value(weights_list[0])
		self._rec_kernel_0 = K.get_value(weights_list[1])
		self._bias_rnn_0 = K.get_value(weights_list[2])
		self._kernel_1 = K.get_value(weights_list[3])
		self._rec_kernel_1 = K.get_value(weights_list[4])
		self._bias_rnn_1 = K.get_value(weights_list[5])
		self._kernel_dense = K.get_value(weights_list[6])
		self._bias_dense = K.get_value(weights_list[7])

		# set up 3 layers
		self.lstm_layer1 = LSTM(n_hidden, return_sequences=True, return_state=True)
		self.lstm_layer2 = LSTM(n_hidden, return_sequences=True, return_state=True)
		self.DenseLayer = TimeDistributed(Dense(n_out, input_shape=(None, 1, self._n_hidden)))

		# set up the model
		inputs = tf.keras.Input(shape=(1, self._n_features))

		# set the first LSTM layer
		LSTMout0, h_state0, c_state0 = self.lstm_layer1(inputs) # , initial_state=self._initial_states[0:2])

		# set the first LSTM layer
		LSTMout1, h_state1, c_state1 = self.lstm_layer2(LSTMout0) #, initial_state=self._initial_states[2:])

		# get the outputs
		outputs = self.DenseLayer(LSTMout1)

		# construct the model
		self.myRNNet = tf.keras.Model(inputs=inputs, outputs=[outputs, h_state0, c_state0, h_state1, c_state1])

		# set up the weights
		# self.myRNNet.set_weights(weights_list)
		self.myRNNet.layers[1].set_weights([self._kernel_0, \
							 self._rec_kernel_0, \
							 self._bias_rnn_0] )

		self.myRNNet.layers[2].set_weights([self._kernel_1, \
							 self._rec_kernel_1, \
							 self._bias_rnn_1] )

		self.myRNNet.layers[3].set_weights([self._kernel_dense, \
							 self._bias_dense ])


	def pred(self, input_strains, states=None):
		"""
		inputs: the (1,1,6) strain input at a single time step
		The model has to be inherite from Model so that it can be called directly.!!!
		"""

		if states is None:
			# set the first LSTM layer
			LSTMout0, h_state0, c_state0 = self.lstm_layer1(input_strains)

			# set the first LSTM layer
			LSTMout1, h_state1, c_state1 = self.lstm_layer2(LSTMout0)

			# get the outputs
			outputs = self.DenseLayer(LSTMout1)

		else:
			# set the first LSTM layer
			LSTMout0, h_state0, c_state0 = self.lstm_layer1(input_strains, initial_state = states[0:2])

			# set the first LSTM layer
			LSTMout1, h_state1, c_state1 = self.lstm_layer2(LSTMout0, initial_state = states[2:])

			# get the outputs
			outputs = self.DenseLayer(LSTMout1)

		# # construct the model
		# self.myRNNet = tf.keras.Model(inputs=inputs, outputs=[outputs, h_state0, c_state0, h_state1, c_state1])

		return outputs, h_state0, c_state0, h_state1, c_state1
