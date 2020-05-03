from keras.utils import plot_model
import sys

import keras
import numpy as np
# import xgboost as xgb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import file_loader

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime
import GAT_model as my_model
from keras.optimizers import Adam
from utils import load_data, preprocess_features
# import models
import argparse

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--dataset', type=str, default='taxi', help='taxi or bike')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of batch')
parser.add_argument('--max_epochs', type=int, default=1000,
                    help='maximum epochs')
parser.add_argument('--att_lstm_num', type=int, default=3,
                    help='the number of time for attention (i.e., value of Q in the paper)')
parser.add_argument('--long_term_lstm_seq_len', type=int, default=3,
                    help='the number of days for attention mechanism (i.e., value of P in the paper)')
parser.add_argument('--short_term_lstm_seq_len', type=int, default=7,
                    help='the length of short term value')
parser.add_argument('--cnn_nbhd_size', type=int, default=3,
                    help='neighbors for local cnn (2*cnn_nbhd_size+1) for area size')
parser.add_argument('--nbhd_size', type=int, default=2,
                    help='for feature extraction')
parser.add_argument('--cnn_flat_size', type=int, default=128,
                    help='dimension of local conv output')
parser.add_argument('--model_name', type=str, default='stdn',
                    help='model name')

args = parser.parse_args()
print(args)


def main():
    # att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype="train",
    #                                                                  att_lstm_num=args.att_lstm_num, \
    #                                                                 long_term_lstm_seq_len=args.long_term_lstm_seq_len,
    #                                                                  short_term_lstm_seq_len=args.short_term_lstm_seq_len, \
    #                                                                  nbhd_size=args.nbhd_size,
    #                                                                  cnn_nbhd_size=args.cnn_nbhd_size)
    """
    print("parameters:")
    print("att_lstm_num= %s" % args.att_lstm_num)
    print("att_lstm_seq_len = %s" % args.long_term_lstm_seq_len)
    print("lstm_seq_len=%s" % len(cnnx))
    print("feature_vec_len=%s" % x.shape[-1])
    print("cnn_flat_size=%s", args.cnn_flat_size)
    print("nbhd_size=%s"% cnnx[0].shape[1])
    print("nbhd_type=%s \n" % cnnx[0].shape[-1])
    """

    modeler = my_model.models()
    """
    model = modeler.stdn(att_lstm_num = args.att_lstm_num,
                        att_lstm_seq_len = args.long_term_lstm_seq_len, \
                         lstm_seq_len = 2, feature_vec_len = 4, \
                          cnn_flat_size = args.cnn_flat_size, nbhd_size = 3, nbhd_type = 2)
    """
    learning_rate = 0.1  # Learning rate for Adam
    model = modeler.stdn(lstm_seq_len=5, feature_vec_len=8)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
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

    plot_model(model, to_file='model.png')
    model.summary()
    model_save_path = "T_GAT.h5"
    # 保存模型
    model.save(model_save_path)
    # 删除当前已存在的模型
    del model


if __name__ == "__main__":
    main()
