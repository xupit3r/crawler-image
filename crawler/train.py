from crawler.models import image_joe
from crawler.review import plot_result, review_dataset
from crawler.gpu import setup_gpus
from dotenv import dotenv_values
import tensorflow as tf
import numpy as np
config = dotenv_values('.env')

tf.get_logger().setLevel('ERROR')

setup_gpus()

def train(options={}, dataset={}, model_name='food'):
  # prefetch (keep memory usage down)
  ds_train = dataset['ds_train'].batch(options['batch_size']).prefetch(buffer_size=options['batch_size'])
  ds_val = dataset['ds_val'].batch(options['batch_size']).prefetch(buffer_size=options['batch_size'])

  if options['show_examples']:
    review_dataset(ds_train)

  model = image_joe(
    num_classes=dataset['num_classes'],
    image_height=options['image_height'],
    image_width=options['image_width'],
    channels=options['channels']
  )

  if options['show_summary']:
    model.summary()

  callbacks = []

  # if we want to save the model, add the callback
  if options['save_model']:
    callbacks.append(
      tf.keras.callbacks.ModelCheckpoint(
        filepath=f'.models/{model_name}.keras',
        save_best_only=True,
        monitor='val_loss'
      )
    )

  # if we want to do some early stopping, add the callback
  if options['early_stop']:
    callbacks.append(
      tf.keras.callbacks.EarlyStopping(
        patience=2
      )
    )

  history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=20,
    callbacks=callbacks
  )

  if options['plot_results']:
    plot_result(history)