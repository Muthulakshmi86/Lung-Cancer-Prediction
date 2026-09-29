import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from Evaluation import evaluation


def Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target):
    X_train = Train_Data.reshape((Train_Data.shape[0], Train_Data.shape[1], 1))
    X_test = Test_Data.reshape((Test_Data.shape[0], Test_Data.shape[1], 1))

    # Define the CNN model
    model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(Train_Target.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()
    TrainX = np.asarray(X_train).astype(np.float32)
    TestX = np.asarray(X_test).astype(np.float32)
    # Train the model
    model.fit(TrainX, Train_Target, epochs=50)
    pred = model.predict(TestX)
    pred = np.reshape(pred, (pred.shape[0], pred.shape[1]))
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred

