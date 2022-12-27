import tensorflow as tf
import pathlib

# reads and prepares a dataset
def prepare_data(url, name):
  data_dir = tf.keras.utils.get_file(
    origin=url,
    fname=name,
    untar=True
  )
  
  data_dir = pathlib.Path(data_dir)
  batch_size = 32
  image_height = 180
  image_width = 180

  ds_train = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
  )

  ds_val = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
  )

  return ds_train, ds_val