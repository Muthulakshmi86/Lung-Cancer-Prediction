import tensorflow as tf
from tensorflow.keras import layers, Model
from Data import *
from keras.callbacks import ModelCheckpoint


def ASPP(inputs, filters):
    b0 = layers.Conv2D(filters, 1, padding="same")(inputs)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation("relu")(b0)

    b1 = layers.Conv2D(filters, 3, padding="same", dilation_rate=6)(inputs)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation("relu")(b1)

    b2 = layers.Conv2D(filters, 3, padding="same", dilation_rate=12)(inputs)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation("relu")(b2)

    b3 = layers.Conv2D(filters, 3, padding="same", dilation_rate=18)(inputs)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation("relu")(b3)

    pool = layers.GlobalAveragePooling2D()(inputs)
    pool = layers.Reshape((1, 1, -1))(pool)
    pool = layers.Conv2D(filters, 1, padding="same")(pool)
    pool = layers.BatchNormalization()(pool)
    pool = layers.Activation("relu")(pool)
    pool = layers.UpSampling2D(size=(inputs.shape[1], inputs.shape[2]), interpolation='bilinear')(pool)

    x = layers.Concatenate()([b0, b1, b2, b3, pool])
    output = layers.Conv2D(filters, 1, padding="same")(x)
    output = layers.BatchNormalization()(output)
    output = layers.Activation("relu")(output)

    return output


# Attention
def cbam_block(feature_map, ratio=8):
    # Channel Attention
    channel_avg = layers.GlobalAveragePooling2D()(feature_map)
    channel_max = layers.GlobalMaxPooling2D()(feature_map)
    shared_dense_one = layers.Dense(feature_map.shape[-1] // ratio, activation='relu')
    shared_dense_two = layers.Dense(feature_map.shape[-1])

    avg_out = shared_dense_two(shared_dense_one(channel_avg))
    max_out = shared_dense_two(shared_dense_one(channel_max))
    channel = layers.Add()([avg_out, max_out])
    channel = layers.Activation('sigmoid')(channel)
    channel = layers.Reshape((1, 1, feature_map.shape[-1]))(channel)
    feature_map = layers.Multiply()([feature_map, channel])

    # Spatial Attention
    avg_pool = tf.reduce_mean(feature_map, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(feature_map, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    spatial = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)

    feature_map = layers.Multiply()([feature_map, spatial])
    return feature_map


# Transformer block
def transformer_block(inputs, num_heads, ff_dim):
    """Transformer block with multi-head attention and feed-forward layers"""
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2


# Encoder
def build_encoder(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    # Backbone (e.g., ResNet50)
    backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    # ASPP applied on the output of the backbone
    aspp_output = ASPP(backbone.output, 256)
    attention = cbam_block(aspp_output)
    x = layers.Reshape((aspp_output.shape[1] * aspp_output.shape[2], aspp_output.shape[3]))(aspp_output)
    transformer_output = transformer_block(x, num_heads=8, ff_dim=512)
    transformer_output = layers.Reshape((aspp_output.shape[1], aspp_output.shape[2], aspp_output.shape[3]))(
        transformer_output)

    return inputs, transformer_output


# Decoder
def build_decoder(encoder_output, skip_connection):
    """DeepLabV3+ style decoder"""
    x = layers.UpSampling2D((4, 4), interpolation="bilinear")(encoder_output)
    x = cbam_block(x)
    # Concatenate with low-level features
    x = layers.Concatenate()([x, skip_connection])
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    output = layers.Conv2D(21, (1, 1), padding="same", activation="softmax")(x)

    return output


# Build the complete model
def DeeplabV3(input_shape=(256, 256, 3)):
    inputs, encoder_output = build_encoder(input_shape)
    # Example of using skip connections from an earlier layer (low-level feature map)
    skip_connection = tf.keras.applications.ResNet50(include_top=False, weights="imagenet",
                                                     input_shape=input_shape).get_layer('conv2_block3_out').output

    decoder_output = build_decoder(encoder_output, skip_connection)

    model = Model(inputs, decoder_output)
    return model


def Model_DeeplabV3(Unet_Path, Image_Path, Mask_Path, Predict_Path, Data, Gt, Target):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(2, Unet_Path, Image_Path, Mask_Path, data_gen_args)

    image_list = os.listdir(os.path.join(Unet_Path, Image_Path))
    image_count = len(image_list)

    testGene = testGenerator(Image_Path, num_image=image_count)
    model = DeeplabV3(Image_Path)
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, hidden_neurons=50, steps_per_epoch=1400, epochs=100, callbacks=[model_checkpoint])
    results = model.predict_generator(testGene, image_count, verbose=1)
    Images = saveResult(Image_Path, results)
    return Images, results
