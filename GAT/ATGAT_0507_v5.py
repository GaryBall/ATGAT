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

    def atgat(self,lstm_seq_len, N=170, lstm_out_size=9,
             optimizer='adagrad', loss='mse', metrics=[]):
        # F = X.shape[1]                # Original feature dimension
        attention_lstm = 3
        F = 3
        F_ = 3
        dropout_rate = 0.1

        graph_inputs = [Input(shape=(F,), name="graph_input_hour_{0}".format(ts + 1)) for ts in range(lstm_seq_len)]
        graph_inputs_day = []

        for att in range(attention_lstm):
            graph_inputs_day.append([Input(shape=(F,),
                                           name="graph_input_day_{0}_{1}".format(att+1, ts+1))
                                     for ts in range(lstm_seq_len)])

        matrix_all = Input(shape=(N,), name="matrix_input_all")
        # local matrix
        # matrix_local = Input(shape=(N,), name="matrix_input_local")
        # distance matrix to find the spatial depency at long distance
        # matrix_dist = Input(shape=(N,), name="matrix_input_dist")
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
            GraphAttention(F_, name='GAT2_local_time0_{0}'.format(ts + 1))([dp_hour_2[ts], matrix_all]) for
            ts in range(lstm_seq_len)]

        # graph_attention_3 = [
        #    GraphAttention(F_, name='GAT2_dist_time0_{0}'.format(ts + 1))([dp_hour_2[ts], matrix_dist]) for
        #     ts in range(lstm_seq_len)]

        # gat_hour_all = [Concatenate(axis=-1)([graph_attention_2[ts], graph_attention_3[ts]]) for
        #            ts in range(lstm_seq_len)]

        hour_vecs = [Dense(units=6, name="hour_dense_time_{0}".format(ts + 1))(graph_attention_2[ts]) for ts in
                     range(lstm_seq_len)]
        hour_vecs = [Activation("relu", name="hour_dense_activation_time_{0}".format(ts + 1))(hour_vecs[ts]) for ts in
                     range(lstm_seq_len)]
        hour_vecs = [Concatenate(axis=-1)([hour_vecs[ts], graph_inputs[ts]]) for ts in range(lstm_seq_len)]

        hour_vec = Concatenate(axis=-1)(hour_vecs)
        hour_vec = Reshape(target_shape = (lstm_seq_len, 9))(hour_vec)
        #lstm
        lstm_hour = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1,
                         recurrent_dropout=0.1,name='lstm_hour')(hour_vec)

        # daily data
        dp_daily_1 = []
        gat_daily1 = []
        dp_daily_2 = []
        gat_daily_2 = []
        gat_daily_3 = []
        gat_daily_all = []
        daily_vecs1 = []
        daily_vecs2 = []
        daily_vecs3 = []
        for att in range(attention_lstm):
            dp_daily_1.append([Dropout(dropout_rate, name="dp1_daily_{0}_{1}".format(att+1, ts+1))(graph_inputs_day[att][ts])
                          for ts in range(lstm_seq_len)])

            gat_daily1.append([GraphAttention(F_, name='GAT1_daily_{0}_{1}'.format(att+1,ts+1))([dp_daily_1[att][ts], matrix_all]) for
                      ts in range(lstm_seq_len)])

            dp_daily_2.append([Dropout(dropout_rate, name="dp2_daily{0}_{1}".format(att+1,ts+1))(gat_daily1[att][ts]) for ts in
                      range(lstm_seq_len)])
            gat_daily_2.append([GraphAttention(F_, name='GAT2_local_daily_{0}_{1}'.format(att+1, ts+1))([dp_daily_2[att][ts], matrix_all]) for
            ts in range(lstm_seq_len)])

            # gat_daily_3.append([
            #     GraphAttention(F_, name='GAT2_dist_daily_{0}_{1}'.format(att+1, ts+1))([dp_daily_2[att][ts], matrix_dist]) for
            #     ts in range(lstm_seq_len)])
            # gat_day2_all.append([Concatenate(axis=-1)([gat_daily_2[att][ts], gat_daily_3[att][ts]]) for
            #        ts in range(lstm_seq_len)])
            daily_vecs1.append([Dense(units=lstm_out_size-3,
                                      name="daily_dense_time_{0}_{1}".format(att+1, ts+1))(gat_daily_2[att][ts])
                                for ts in range(lstm_seq_len)])
            daily_vecs2.append([Activation("relu",
                                           name="daily_dense_activation_time_{0}_{1}".format(att+1,ts+1))(daily_vecs1[att][ts])
                                for ts in range(lstm_seq_len)])

            daily_vecs3.append([Concatenate(axis=-1)([daily_vecs2[att][ts], graph_inputs_day[att][ts]])
                                for ts in range(lstm_seq_len)])

        daily_vec = [Concatenate(axis=-1)(daily_vecs3[att]) for att in range(attention_lstm)]
        daily_vec1 = [Reshape(target_shape=(lstm_seq_len, lstm_out_size))(daily_vec[att]) for att in range(attention_lstm)]

        # att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1,
        #                   name="att_lstm_{0}".format(att + 1))(daily_vec1[att]) for att in range(attention_lstm)]

        att_low_level = [Attention(method='cba')([daily_vec1[att], lstm_hour]) for att in range(attention_lstm)]
        att_low_level = Concatenate(axis=-1)(att_low_level)
        att_low_level = Reshape(target_shape=(attention_lstm, lstm_out_size))(att_low_level)

        att_high_level = LSTM(units=6, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(
            att_low_level)

        lstm_all = Concatenate(axis=-1)([lstm_hour, att_high_level])

        act_value = Activation('tanh')(lstm_all)
        output = Dense(units=3)(act_value)
        input = graph_inputs
        for i in range(attention_lstm):
            input = input+graph_inputs_day[i]
        input = input + [matrix_all, ]
        # input = input + [matrix_all, ] + [matrix_local, ] + [matrix_dist, ]

        model = Model(inputs=input,
                      outputs=output)
        model.compile(optimizer = optimizer, loss=loss, metrics=metrics)
        return model
