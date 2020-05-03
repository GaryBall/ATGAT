from __future__ import print_function
from keras import backend as K
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from GCN.graph  import GraphConvolution
from GCN.utils  import *
from keras.utils import plot_model

import time

# Define parameters
DATASET = 'cora'  # 数据集的名称
FILTER = 'localpool'  # 'chebyshev' 采用的卷积类型
MAX_DEGREE = 2  # 最大多项式的度
SYM_NORM = True  # 是否对称正则化
NB_EPOCH = 200  # epoches的数量
PATIENCE = 10  # early stopping patience

# shape为形状元组，不包括batch_size
# 例如shape=(32, )表示预期的输入将是一批32维的向量
feature_len = 3
dropout_rate = 0
lstm_seq_len = 5
lstm_out_size = 3
N = 170

"""
X_in = Input(shape=(X.shape[1],))
# 定义模型架构
# 注意：我们将图卷积网络的参数作为张量列表传递
# 更优雅的做法需要重写Layer基类
H = Dropout(0.5)(X_in)
print(K.int_shape(H))
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + G)
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H] + G)
"""



graph_inputs = [Input(shape=(feature_len,), name="graph_input_time_{0}".format(ts + 1)) for ts in range(lstm_seq_len)]
matrix_in = Input(shape=(N,), name="matrix_input_")

# lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")
dropout1 = [Dropout(dropout_rate, name="dropout1_time0_{0}".format(ts + 1))(graph_inputs[ts]) for ts in range(lstm_seq_len)]

graph_attention_1 = [GraphConvolution(feature_len, name='GCN1_time0_{0}'.format(ts + 1))([dropout1[ts], matrix_in]) for
                             ts in range(lstm_seq_len)]

dropout2 = [Dropout(dropout_rate, name="dropout2_time0_{0}".format(ts + 1))(graph_attention_1[ts]) for ts in
                    range(lstm_seq_len)]
graph_attention_2 = [GraphConvolution(feature_len, name='GCN2_time0_{0}'.format(ts + 1))([dropout2[ts], matrix_in]) for
                             ts in range(lstm_seq_len)]

dropout3 = [Dropout(dropout_rate, name="dropout3_time0_{0}".format(ts + 1))(graph_attention_2[ts]) for ts in
                    range(lstm_seq_len)]
graph_attention_3 = [GraphConvolution(feature_len, name='GCN3_time0_{0}'.format(ts + 1))([dropout3[ts], matrix_in]) for
                             ts in range(lstm_seq_len)]

dropout4 = [Dropout(dropout_rate, name="dropout4_time0_{0}".format(ts + 1))(graph_attention_3[ts]) for ts in
                    range(lstm_seq_len)]
graph_attention_4 = [GraphConvolution(feature_len, name='GCN4_time0_{0}'.format(ts + 1))([dropout4[ts], matrix_in]) for
                             ts in range(lstm_seq_len)]

dropout5 = [Dropout(dropout_rate, name="dropout5_time0_{0}".format(ts + 1))(graph_attention_4[ts]) for ts in
                    range(lstm_seq_len)]
graph_attention_5 = [GraphConvolution(feature_len, name='GCN5_time0_{0}'.format(ts + 1))([dropout5[ts], matrix_in]) for
                             ts in range(lstm_seq_len)]


nbhd_vecs = [Dense(units = 8, name = "nbhd_dense_time_{0}".format(ts+1))(graph_attention_5[ts]) for ts in range(lstm_seq_len)]
nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(lstm_seq_len)]

nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
nbhd_vec = Reshape(target_shape = (lstm_seq_len, 8))(nbhd_vec)
# lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])
print(K.int_shape(nbhd_vec))
#lstm
lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(nbhd_vec)
output = Dense(units=3)(lstm)
# Compile model
model = Model(inputs=graph_inputs + [matrix_in, ], outputs=output)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
model_save_path = "T_GCN.h5"
model.save(model_save_path)
plot_model(model, to_file='model.png')

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999


"""
# Fit
for epoch in range(1, NB_EPOCH + 1):
    # 统计系统时钟的时间戳
    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,  # 向sample_weight参数传递train_mask用于样本掩码
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
    # 预测模型在整个数据集上的输出
    preds = model.predict(graph, batch_size=A.shape[0])
    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))


"""