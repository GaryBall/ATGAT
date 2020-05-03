import pickle
import numpy as np
import json
import tensorflow as tf
import keras
from GAT_ori import GraphAttention
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Input, Reshape, Dropout, Concatenate, LSTM
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import ipdb


class baselines:
    def __init__(self):
        pass


class models:
    def __init__(self):
        pass

    def stdn(self,lstm_seq_len, N=170, lstm_out_size=9,
             optimizer='adagrad', loss='mse', metrics=[]):
        # F = X.shape[1]                # Original feature dimension
        F = 3
        F_ = 3
        dropout_rate = 0
        graph_inputs = [Input(shape=(F,), name="graph_input_hour_{0}".format(ts + 1)) for ts in range(lstm_seq_len)]
        graph_inputs_day = [Input(shape=(F,), name="graph_input_daily_{0}".format(ts + 1)) for ts in
                            range(lstm_seq_len)]
        matrix_all = Input(shape=(N,), name="matrix_input_all")
        # local matrix
        matrix_local = Input(shape=(N,), name="matrix_input_local")
        # distance matrix to find the spatial depency at long distance
        matrix_dist = Input(shape=(N,), name="matrix_input_dist")
        # lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")
        # print(K.int_shape(lstm_inputs))

        # hourly data
        dp_hour_1 = [Dropout(dropout_rate, name="dp1_hour_time0_{0}".format(ts + 1))(graph_inputs[ts]) for ts in
                     range(lstm_seq_len)]

        graph_attention_1 = [GraphAttention(F_, name='GAT1_time0_{0}'.format(ts + 1))([dp_hour_1[ts], matrix_all]) for
                             ts in range(lstm_seq_len)]

        dp_hour_2 = [Dropout(dropout_rate, name="dp2_hour_time0_{0}".format(ts + 1))(graph_attention_1[ts]) for ts in
                     range(lstm_seq_len)]

        graph_attention_2 = [
            GraphAttention(F_, name='GAT2_local_time0_{0}'.format(ts + 1))([dp_hour_2[ts], matrix_local]) for
            ts in range(lstm_seq_len)]

        graph_attention_3 = [
            GraphAttention(F_, name='GAT2_dist_time0_{0}'.format(ts + 1))([dp_hour_2[ts], matrix_dist]) for
            ts in range(lstm_seq_len)]

        gat_hour_all = [Concatenate(axis=-1)([graph_attention_2[ts], graph_attention_3[ts]]) for
                   ts in range(lstm_seq_len)]

        hour_vecs = [Dense(units=9, name="hour_dense_time_{0}".format(ts + 1))(gat_hour_all[ts]) for ts in
                     range(lstm_seq_len)]
        hour_vecs = [Activation("relu", name="hour_dense_activation_time_{0}".format(ts + 1))(hour_vecs[ts]) for ts in
                     range(lstm_seq_len)]
        hour_vecs = [Concatenate(axis=-1)([hour_vecs[ts], graph_inputs[ts]]) for ts in range(lstm_seq_len)]

        hour_vec = Concatenate(axis=-1)(hour_vecs)
        hour_vec = Reshape(target_shape = (lstm_seq_len, 12))(hour_vec)
        #lstm
        lstm_hour = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1,
                         recurrent_dropout=0.1,name='lstm_hour')(hour_vec)

        # daily data
        dp_daily_1 = [Dropout(dropout_rate, name="dp1_daily0_{0}".format(ts + 1))(graph_inputs_day[ts]) for ts in
                    range(lstm_seq_len)]

        gat_daily1 = [GraphAttention(F_, name='GAT1_daily0_{0}'.format(ts + 1))([dp_daily_1[ts], matrix_all]) for
                             ts in range(lstm_seq_len)]

        dp_daily_2 = [Dropout(dropout_rate, name="dp2_daily0_{0}".format(ts + 1))(gat_daily1[ts]) for ts in
                      range(lstm_seq_len)]

        gat_daily_2 = [
            GraphAttention(F_, name='GAT2_local_daily0_{0}'.format(ts + 1))([dp_daily_2[ts], matrix_local]) for
            ts in range(lstm_seq_len)]

        gat_daily_3 = [
            GraphAttention(F_, name='GAT2_dist_daily0_{0}'.format(ts + 1))([dp_daily_2[ts], matrix_dist]) for
            ts in range(lstm_seq_len)]

        gat_daily_all = [Concatenate(axis=-1)([gat_daily_2[ts], gat_daily_3[ts]]) for
                   ts in range(lstm_seq_len)]

        daily_vecs = [Dense(units=9, name="daily_dense_time_{0}".format(ts + 1))(gat_daily_all[ts]) for ts in
                     range(lstm_seq_len)]
        daily_vecs = [Activation("relu", name="daily_dense_activation_time_{0}".format(ts + 1))(daily_vecs[ts]) for ts in
                     range(lstm_seq_len)]
        daily_vecs = [Concatenate(axis=-1)([daily_vecs[ts], graph_inputs_day[ts]]) for ts in range(lstm_seq_len)]

        daily_vec = Concatenate(axis=-1)(daily_vecs)
        daily_vec = Reshape(target_shape=(lstm_seq_len, 12))(daily_vec)

        lstm_daily = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,
                          name='lstm_daily')(daily_vec)

        lstm_all = Concatenate(axis=-1)([lstm_hour, lstm_daily])

        act_value = Activation('tanh')(lstm_all)
        output = Dense(units=3)(act_value)

        # model = Model(inputs = graph_inputs + [matrix_in,] + [lstm_inputs,], outputs = lstm)
        model = Model(inputs=graph_inputs + graph_inputs_day + [matrix_all, ] + [matrix_local, ] + [matrix_dist, ],
                      outputs=output)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model
