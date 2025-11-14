import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model

def build_unet(input_shape=(256, 256, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    # Encoder
    x1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x1 = Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
    p1 = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    x2 = Conv2D(128, (3, 3), activation="relu", padding="same")(x2)
    p2 = MaxPooling2D((2, 2))(x2)

    # Decoder
    u1 = UpSampling2D((2, 2))(p2)
    u1 = Conv2D(128, (3, 3), activation="relu", padding="same")(u1)
    merge1 = Concatenate()([x2, u1])

    u2 = UpSampling2D((2, 2))(merge1)
    u2 = Conv2D(64, (3, 3), activation="relu", padding="same")(u2)
    merge2 = Concatenate()([x1, u2])

    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(merge2)
    
    return Model(inputs, outputs)

unet_model = build_unet()
unet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
unet_model.summary()
