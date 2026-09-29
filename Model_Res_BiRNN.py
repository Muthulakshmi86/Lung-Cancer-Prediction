import numpy as np
import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras.models import Model
from Evaluation import evaluation
from keras.layers import Bidirectional, SimpleRNN, GlobalAveragePooling1D, Reshape, Dense, Multiply, Add, Activation


def resblock(inputs, filters, strides):
    # Main path
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        strides=strides,
        padding='same',
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding='same',
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    y = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=1,
        strides=strides,
        padding='same',
    )(inputs)
    y = tf.keras.layers.BatchNormalization()(y)

    # Concatenate paths
    x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.ReLU()(x)

    return x


def multilevel_context_attention(inputs, sol, act, reduction_ratio=8):
    channel = inputs.shape[-1]
    shared_dense_one = Dense(channel // reduction_ratio, activation=act[sol[2]])
    shared_dense_two = Dense(channel)
    avg_pool = GlobalAveragePooling1D()(inputs)
    avg_pool = Reshape((1, channel))(avg_pool)
    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)
    max_out = shared_dense_two(shared_dense_one(max_pool))
    channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
    channel_refined = Multiply()([inputs, channel_attention])

    return channel_refined


def Model_Res_BiRNN(Train_Data, Train_Target, Test_data, Test_Target, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]
    act = ['linear', 'relu', 'softmax', 'sigmoid', 'tanh', 'relu']
    trainX = np.reshape(Train_Data, (Train_Data.shape[0], 1, Train_Data.shape[1]))
    testX = np.reshape(Test_data, (Test_data.shape[0], 1, Test_data.shape[1]))
    hidden_units = 64
    inputs = tf.keras.Input(shape=(1, trainX.shape[2]))

    #  residual block before RNN
    x = resblock(inputs, filters=64, strides=1)
    # Context Attention Layer
    x = multilevel_context_attention(x, sol, act)
    x = Bidirectional(SimpleRNN(sol[0]))(x)
    outputs = Dense(Train_Target.shape[1], activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=sol[1]), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(trainX, Train_Target,
              batch_size=4,
              epochs=50, steps_per_epoch=100)
    pred = model.predict(testX)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred



