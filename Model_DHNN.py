import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from Evaluation import evaluation


def Model_DHNN(Train_Data, Train_Target, Test_Data, Test_Target):
    X_train = Train_Data.reshape((Train_Data.shape[0], Train_Data.shape[1], 1))
    X_test = Test_Data.reshape((Test_Data.shape[0], Test_Data.shape[1], 1))
    input_layer = Input(shape=(X_train.shape[1], 1))
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(Train_Target.shape[1], activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Train_Target, epochs=100, batch_size=32)
    pred = model.predict(X_test)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred

