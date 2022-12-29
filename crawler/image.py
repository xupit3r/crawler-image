from crawler.models import food_model, food_model_summary
from crawler.datasets import prepare_data
from crawler.review import plot_result, review_dataset
from crawler.gpu import setup_gpus
from dotenv import dotenv_values
import tensorflow as tf
import numpy as np

config = dotenv_values('.env')

tf.get_logger().setLevel('ERROR')

setup_gpus()

def train(show_summary=False, save_model=False, show_examples=False, early_stop=False, plot_results=False):
  ds_train, ds_val = prepare_data(
    url=config['DATASET_URL'],
    name=config['DATASET_NAME']
  )

  if show_summary:
    food_model_summary()

  if show_examples:
    review_dataset(ds_train)

  num_classes = len(ds_train.class_names)
  ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)
  ds_val = ds_val.prefetch(buffer_size=tf.data.AUTOTUNE)

  model = food_model(num_classes)

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

  history = model.fit(
    ds_train,
    epochs=100,
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