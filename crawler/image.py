from crawler.models import pizza_model
from crawler.datasets import prepare_data
from crawler.review import plot_result
from crawler.gpu import setup_gpus
import tensorflow as tf
from dotenv import dotenv_values

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