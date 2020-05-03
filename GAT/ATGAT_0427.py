import pickle
import numpy as np
import json
import tensorflow as tf
import keras
from GAT_ori import GraphAttention
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Input, Reshape, Dropout, Concatenate, LSTM
from attention import Attention
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import ipdb


class baselines:
    def __init__(self):
        pass


class models:
    def __init__(self):
        pass

    def atgat(self,lstm_seq_len, N=307, lstm_out_size=9,
             optimizer='adagrad', loss='mse', metrics=[]):
        # F = X.shape[1]                # Original feature dimension
        attention_lstm = 2
        F = 3
        F_ = 3
        dropout_rate = 0
        graph_inputs = [Input(shape=(F,), name="graph_input_hour_{0}".format(ts + 1)) for ts in range(lstm_seq_len)]

        graph_inputs_day1 = [Input(shape=(F,), name="graph_input_day1_{0}".format(ts + 1)) for ts in
                            range(lstm_seq_len)]

        graph_inputs_day2 = [Input(shape=(F,), name="graph_input_day2_{0}".format(ts + 1)) for ts in
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

        # daily data 1
        dp_d1_1 = [Dropout(dropout_rate, name="dp1_day1_{0}".format(ts + 1))(graph_inputs_day1[ts]) for ts in
                      range(lstm_seq_len)]

        gat_d1_1 = [GraphAttention(F_, name='GAT1_day1_{0}'.format(ts + 1))([dp_d1_1 [ts], matrix_all]) for
                      ts in range(lstm_seq_len)]

        dp_d1_2 = [Dropout(dropout_rate, name="dp2_d1_{0}".format(ts + 1))(gat_d1_1[ts]) for ts in
                      range(lstm_seq_len)]

        gat_d1_2 = [
            GraphAttention(F_, name='GAT2_local_d1_{0}'.format(ts + 1))([dp_d1_2[ts], matrix_local]) for
            ts in range(lstm_seq_len)]

        gat_d1_3 = [
            GraphAttention(F_, name='GAT2_dist_d1_{0}'.format(ts + 1))([gat_d1_2[ts], matrix_dist]) for
            ts in range(lstm_seq_len)]

        gat_d1_all = [Concatenate(axis=-1)([gat_d1_2[ts], gat_d1_3[ts]]) for
                         ts in range(lstm_seq_len)]

        d1_vecs = [Dense(units=lstm_out_size, name="d1_dense_time_{0}".format(ts + 1))(gat_d1_all[ts]) for ts in
                      range(lstm_seq_len)]
        d1_vecs = [Activation("relu", name="d1_dense_activation_time_{0}".format(ts + 1))(d1_vecs[ts]) for ts
                      in range(lstm_seq_len)]
        d1_vecs = [Concatenate(axis=-1)([d1_vecs[ts], graph_inputs_day1[ts]]) for ts in range(lstm_seq_len)]
        d1_vec = Concatenate(axis=-1)(d1_vecs)
        d1_vec = Reshape(target_shape=(lstm_seq_len, lstm_out_size+F))(d1_vec)

        # lstm_daily = LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1,
        #                   name='lstm_daily')(daily_vec)


        # daily data 2
        dp_d2_1 = [Dropout(dropout_rate, name="dp1_day2_{0}".format(ts + 1))(graph_inputs_day2[ts]) for ts in
                   range(lstm_seq_len)]

        gat_d2_1 = [GraphAttention(F_, name='GAT1_day2_{0}'.format(ts + 1))([dp_d2_1[ts], matrix_all]) for
                    ts in range(lstm_seq_len)]

        dp_d2_2 = [Dropout(dropout_rate, name="dp2_d2_{0}".format(ts + 1))(gat_d2_1[ts]) for ts in
                   range(lstm_seq_len)]

        gat_d2_2 = [
            GraphAttention(F_, name='GAT2_local_d2_{0}'.format(ts + 1))([dp_d2_2[ts], matrix_local]) for
            ts in range(lstm_seq_len)]

        gat_d2_3 = [
            GraphAttention(F_, name='GAT2_dist_d2_{0}'.format(ts + 1))([gat_d2_2[ts], matrix_dist]) for
            ts in range(lstm_seq_len)]

        gat_d2_all = [Concatenate(axis=-1)([gat_d2_2[ts], gat_d2_3[ts]]) for
                      ts in range(lstm_seq_len)]

        d2_vecs = [Dense(units=lstm_out_size, name="d2_dense_time_{0}".format(ts + 1))(gat_d2_all[ts]) for ts in
                   range(lstm_seq_len)]
        d2_vecs = [Activation("relu", name="d2_dense_activation_time_{0}".format(ts + 1))(d2_vecs[ts]) for ts
                   in range(lstm_seq_len)]
        d2_vecs = [Concatenate(axis=-1)([d2_vecs[ts], graph_inputs_day2[ts]]) for ts in range(lstm_seq_len)]

        d2_vec = Concatenate(axis=-1)(d2_vecs)
        d2_vec = Reshape(target_shape=(lstm_seq_len, lstm_out_size+F))(d2_vec)
        print(K.int_shape(d2_vec))

        att1_low_level = Attention(method='cba')([d1_vec, lstm_hour])
        att2_low_level = Attention(method='cba')([d2_vec, lstm_hour])

        att_low_level_list = [att1_low_level, att2_low_level]
        att_low_level = Concatenate(axis=-1)(att_low_level_list)
        print(K.int_shape(att_low_level))
        att_low_level = Reshape(target_shape=(attention_lstm, lstm_out_size+F))(att_low_level)

        att_high_level = LSTM(units=lstm_out_size,
                              return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        lstm_all = Concatenate(axis=-1)([lstm_hour, att_high_level])

        act_value = Activation('tanh')(lstm_all)
        output = Dense(units=3)(act_value)
        model = Model(inputs=graph_inputs + graph_inputs_day1+graph_inputs_day2 # + attention_lstm_input
                             + [matrix_all, ] + [matrix_local, ] + [matrix_dist, ],
                      outputs=output)
        model.compile(optimizer = optimizer, loss=loss, metrics=metrics)
        return model
