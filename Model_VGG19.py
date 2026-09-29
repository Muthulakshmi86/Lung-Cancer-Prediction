import numpy as np
from keras.applications.vgg19 import VGG19
from keras import layers, models, backend as K


def VGG_19(Target):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False
    # input layer
    inputs = layers.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    # Add layers
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(Target.shape[1], activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Model_VGG19(Image, Target):
    IMG_SIZE = 32
    Train_X = np.array([np.resize(img, (IMG_SIZE, IMG_SIZE, 3)) for img in Image])
    model = VGG_19(Target)
    model.summary()
    model.fit(Train_X, Target, epochs=2)
    layer_outputs = [layer.output for layer in model.layers]
    functors = [K.function([model.input], [out]) for out in layer_outputs]
    layerNo = 6
    Feats = []
    for i in range(Train_X.shape[0]):
        test = Train_X[i][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)

    return np.asarray(Feats)
