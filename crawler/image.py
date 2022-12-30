from crawler.models import image_joe
from crawler.datasets import prepare_data
from crawler.review import plot_result, review_dataset
from crawler.gpu import setup_gpus
from dotenv import dotenv_values
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

config = dotenv_values('.env')

tf.get_logger().setLevel('ERROR')

setup_gpus()

def convert_data(image, label, num_classes):
  return (tf.cast(image, tf.float32) / 255., tf.one_hot(label, num_classes))

def train(show_summary=False, save_model=False, show_examples=False, early_stop=False, plot_results=False, local_ds=False):
  batch_size = 128
  image_height = 28
  image_width = 28

  if local_ds:
    ds_train, ds_val = prepare_data(
      url=config['DATASET_URL'],
      name=config['DATASET_NAME'],
      batch_size=batch_size,
      image_height=image_height,
      image_width=image_width
    )

    num_classes = len(ds_train.class_names)
  else:
    [ds_train, ds_val], info = tfds.load(
      'fashion_mnist',
      split=['train', 'test'], 
      shuffle_files=True,
      as_supervised=True,
      with_info=True
    )

    num_classes = info.features['label'].num_classes
    ds_train = ds_train.map(lambda image, label: convert_data(image, label, num_classes))
    ds_val = ds_val.map(lambda image, label: convert_data(image, label, num_classes))

  # prefetch (keep memory usage down)
  ds_train = ds_train.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
  ds_val = ds_val.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  if show_examples:
    review_dataset(ds_train)

  model = image_joe(
    num_classes=num_classes,
    image_height=28,
    image_width=28,
    channels=1
  )

  if show_summary:
    model.summary()

  callbacks = []

  # if we want to save the model, add the callback
  if save_model:
    callbacks.append(
      tf.keras.callbacks.ModelCheckpoint(
        filepath='models/food.keras',
        save_best_only=True,
        monitor='val_loss'
      )
    )

  # if we want to do some early stopping, add the callback
  if early_stop:
    callbacks.append(
      tf.keras.callbacks.EarlyStopping(
        patience=2
      )
    )

  if local_ds:
    history = model.fit(
      ds_train,
      validation_data=ds_val,
      epochs=100,
      batch_size=batch_size,
      callbacks=callbacks
    )
  else:
    history = model.fit(
      ds_train,
      validation_data=ds_val,
      epochs=20,
      batch_size=batch_size,
      callbacks=callbacks
    )

  if plot_results:
    plot_result(history)

def load_model(model=None):
  try:
    model = tf.keras.models.load_model(f'models/{model}.keras')
    return model
  except IOError as err:
    print(f'failed to load model {model} --- {err}')
  
  return False

def pizza_predict(image):
  CLASSES = ['not pizza', 'pizza']
  model = load_model('pizza')
  predictions = model.predict(image)
  class_index = np.argmax(predictions[0])
  return CLASSES[class_index]