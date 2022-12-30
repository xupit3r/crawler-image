from tensorflow import keras
from tensorflow.keras import layers, regularizers

# classifies pizza and not pizza
def pizza_model(num_classes):
  model = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.Rescaling(1./255),
    layers.Conv2D(32,
    3,
    activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(256, 3, activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='sigmoid')
  ])

  model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
  )

  return model


# an image classifier
def image_joe(num_classes=1, image_height=128, image_width=128, channels=3):
  # inputs
  inputs = keras.Input(name='image', shape=(image_height, image_width, channels))

  # block 1
  # rescaling = layers.Rescaling(1./255)(inputs)

  # block 2
  conv_1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
  batched_1 = layers.BatchNormalization()(conv_1)

  # block 3
  conv_2 = layers.Conv2D(128, 3, padding='same', activation='relu')(batched_1)
  batched_2 = layers.BatchNormalization()(conv_2)

  # block 4
  conv_5 = layers.Conv2D(256, 3, padding='same', activation='relu')(batched_2)
  conv_6 = layers.Conv2D(256, 3, padding='same', activation='relu')(conv_5)
  pool_1 = layers.MaxPooling2D((2,2), padding='same')(conv_6)

  # block 5
  flatten = layers.Flatten()(pool_1)
  dense_1 = layers.Dense(
    2048,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.002)
  )(flatten)
  dropout_1 = layers.Dropout(0.5)(dense_1)
  dense_2 = layers.Dense(
    2048,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.002)
  )(dropout_1)
  dropout_2 = layers.Dropout(0.25)(dense_2)
  dense_3 = layers.Dense(
    2048,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.002)
  )(dropout_2)

  # outputs
  outputs = layers.Dense(num_classes, activation='softmax')(dense_3)

  model = keras.Model(
    inputs=inputs,
    outputs=outputs
  )

  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=10e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  return model
