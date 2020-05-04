from keras.utils import plot_model
import sys

import keras
import numpy as np
# import xgboost as xgb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime
import GAT.ATGAT_0502_v3 as my_model
# import test_models as my_model
from keras.optimizers import Adam
from utils import load_data, preprocess_features
# import models
import argparse


def main():

    modeler = my_model.models()

    """
    model = modeler.stdn(att_lstm_num = args.att_lstm_num,
                         att_lstm_seq_len = args.long_term_lstm_seq_len,
                         lstm_seq_len = 2, feature_vec_len = 4,
                         cnn_flat_size = args.cnn_flat_size, nbhd_size = 3, nbhd_type = 2)
    """
    learning_rate = 0.1  # Learning rate for Adam

    model = modeler.atgat(lstm_seq_len=5, N=170)

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  weighted_metrics=['mse'])


    """
    # Callbacks
    es_patience = 100
    N = 815
    es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
    tb_callback = TensorBoard(batch_size=N)
    mc_callback = ModelCheckpoint('logs/best_model.h5',
                                  monitor='val_weighted_acc',
                                  save_best_only=True,
                                  save_weights_only=True)

    # Train model
    
    epochs = 500  # Number of training epochs
    A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = load_data('cora')
    print("parameters:")
    print(X.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    print(Y_val.shape)
    print(idx_train.shape)
    print(idx_val.shape)
    print(idx_test.shape)
    print(A.shape)
    print('end\n')

    validation_data = ([X, A], Y_val, idx_val)
    model.fit([X, A],
              Y_train,
              sample_weight=idx_train,
              epochs=epochs,
              batch_size=N,
              validation_data=validation_data,
              shuffle=False,  # Shuffling data means shuffling the whole graph
              callbacks=[es_callback, tb_callback, mc_callback])

    # Load best model
    model.load_weights('logs/best_model.h5')

    # Evaluate model
    eval_results = model.evaluate([X, A],
                                  Y_test,
                                  sample_weight=idx_test,
                                  batch_size=N,
                                  verbose=0)
                                  
    print('Done.\n'
          'Test loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))
    """

    plot_model(model, to_file='h5_models/ATGAT_0502.png')
    model.summary()
    model_save_path = "h5_models/ATGAT_0502.h5"
    # 保存模型
    model.save(model_save_path)
    # 删除当前已存在的模型
    del model


if __name__ == "__main__":
    main()
