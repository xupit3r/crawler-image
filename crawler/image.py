from crawler.models import pizza_model
from crawler.datasets import prepare_data
from crawler.review import plot_result
from crawler.gpu import setup_gpus
from dotenv import dotenv_values
import tensorflow as tf
import numpy as np

config = dotenv_values('.env')

tf.get_logger().setLevel('ERROR')

setup_gpus()

def train(save_model=False, plot_results=False):
  ds_train, ds_val = prepare_data(
    url=config['DATASET_URL'],
    name=config['DATASET_NAME']
  )

  num_classes = 2
  ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)
  ds_val = ds_val.prefetch(buffer_size=tf.data.AUTOTUNE)

  model = pizza_model(num_classes)

  if save_model:
    callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
        filepath='models/pizza.keras',
        save_best_only=True,
        monitor='val_loss'
      )
    ]
  else:
    callbacks = None

  history = model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_val,
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