import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

from Evaluation import evaluation  # You must provide this or replace it with standard metrics.


def mask_rcnn_model(input_shape=(256, 256, 3), num_classes=21):
    """
    Custom Mask R-CNN model with ResNet50 backbone, RPN, and mask head.
    """
    # Feature extractor
    backbone = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    backbone.trainable = False
    feature_map = backbone.output

    # Region Proposal Network (RPN)
    rpn = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(feature_map)
    rpn_class = layers.Conv2D(9 * 2, (1, 1), activation='sigmoid')(rpn)
    rpn_bbox = layers.Conv2D(9 * 4, (1, 1), activation='linear')(rpn)

    # ROI Pooling (approximated)
    pooled_features = layers.GlobalAveragePooling2D()(feature_map)
    dense_feat = layers.Reshape((1, 1, -1))(pooled_features)

    # Fully connected head for classification and bbox regression
    fc1 = layers.Dense(1024, activation='relu')(dense_feat)

    class_logits = layers.Dense(num_classes, activation='softmax')(fc1)

    # Mask head
    mask_conv1 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(feature_map)
    mask_conv2 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(mask_conv1)
    mask_deconv = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(mask_conv2)
    mask_output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name='mask_output')(mask_deconv)

    # Final model
    model = models.Model(inputs=backbone.input, outputs=class_logits)
    return model


def Model_RCNN(Train_Data, Train_target_cls, Test_Data, Test_target_cls):
    IMG_SIZE = 256
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    num_classes = Train_target_cls.shape[1]

    model = mask_rcnn_model(input_shape=(256, 256, 3), num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss= 'categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    Tra = Train_target_cls.reshape((Train_target_cls.shape[0], 1, 1, Train_target_cls.shape[1]))
    model.fit(Train_X, Tra, epochs=20)
    preds = model.predict(Test_X)
    avg = (np.min(preds) + np.max(preds)) / 2
    preds[preds >= avg] = 1
    preds[preds < avg] = 0
    Eval = evaluation(Test_target_cls, preds)

    return Eval, preds

