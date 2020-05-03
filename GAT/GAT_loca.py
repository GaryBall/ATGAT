import pickle
import numpy as np
import json
import tensorflow as tf
import keras
from GAT_ori import GraphAttention
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import ipdb
import attention


class baselines:
    def __init__(self):
        pass


class models:
    def __init__(self):
        pass

    def stdn(self,lstm_seq_len,  N = 307, lstm_out_size = 9,\
    optimizer = 'adagrad', loss = 'mse', metrics=[]):
        # F = X.shape[1]                # Original feature dimension
        F = 3
        F_ = 3
        dropout_rate = 0
        graph_inputs = [Input(shape=(F,), name="graph_input_time_{0}".format(ts + 1)) for ts in range(lstm_seq_len)]
        matrix_local = Input(shape=(N,), name="matrix_input_local")
        # distance matrix to find the spatial depency at long distance
        matrix_dist = Input(shape=(N,), name="matrix_input_dist")
        # lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")
        # print(K.int_shape(lstm_inputs))

        dp_local_1 = [Dropout(dropout_rate, name="dp_local1_time0_{0}".format(ts + 1))(graph_inputs[ts]) for ts in range(lstm_seq_len)]

        graph_attention_1 = [GraphAttention(F_, name='GAT1_local1_time0_{0}'.format(ts + 1))([dp_local_1[ts], matrix_local]) for
                             ts in range(lstm_seq_len)]

        dp_local_2 = [Dropout(dropout_rate, name="dp_local2_time0_{0}".format(ts + 1))(graph_attention_1[ts]) for ts in
                    range(lstm_seq_len)]
        graph_attention_2 = [GraphAttention(F_, name='GAT2_local2_time0_{0}'.format(ts + 1))([dp_local_2[ts], matrix_local]) for
                             ts in range(lstm_seq_len)]
        local_vecs = [Dense(units = 9, name = "local_dense_time_{0}".format(ts+1))(graph_attention_2[ts]) for ts in range(lstm_seq_len)]
        local_vecs = [Activation("relu", name = "local_activation_time_{0}".format(ts+1))(local_vecs[ts]) for ts in range(lstm_seq_len)]



        dp_dist_1= [Dropout(dropout_rate, name="dp_dis1_time0_{0}".format(ts + 1))(graph_inputs[ts]) for ts in
                    range(lstm_seq_len)]

        ga_dis_1 = [GraphAttention(F_, name='GAT_dis1_time0_{0}'.format(ts + 1))([dp_dist_1[ts], matrix_dist]) for
                             ts in range(lstm_seq_len)]

        dp_dist_2 = [Dropout(dropout_rate, name="dp_dis2_time0_{0}".format(ts + 1))(ga_dis_1[ts]) for ts in
                    range(lstm_seq_len)]
        ga_dis_2 = [GraphAttention(F_, name='GAT_dis2_time0_{0}'.format(ts + 1))([dp_dist_2[ts], matrix_local]) for
                             ts in range(lstm_seq_len)]
        dis_vecs = [Dense(units=9, name="dis_dense_time_{0}".format(ts + 1))(ga_dis_2[ts]) for ts in
                      range(lstm_seq_len)]
        dis_vecs = [Activation("relu", name="dis_activation_time_{0}".format(ts + 1))(dis_vecs[ts]) for ts in
                      range(lstm_seq_len)]

        comb_vecs = [Concatenate(axis=-1)([local_vecs[ts], graph_inputs[ts],dis_vecs[ts]]) for ts in range(lstm_seq_len)]

        nbhd_vec = Concatenate(axis=-1)(comb_vecs)
        nbhd_vec = Reshape(target_shape = (lstm_seq_len, 21))(nbhd_vec)
        # lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])
        #lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(nbhd_vec)
        pred_volume = Activation('tanh')(lstm)
        output = Dense(units=3)(pred_volume)


        # model = Model(inputs = graph_inputs + [matrix_in,] + [lstm_inputs,], outputs = lstm)
        model = Model(inputs=graph_inputs + [matrix_local, ] + [matrix_dist, ], outputs=output)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model
