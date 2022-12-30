import tensorflow as tf
import pathlib

# reads and prepares a dataset
def prepare_data(url, name, batch_size=32, image_height=128, image_width=128):
  data_dir = tf.keras.utils.get_file(
    origin=url,
    fname=name,
    untar=True
  )
  
  data_dir = pathlib.Path(data_dir)

  ds_train = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
  )
  

  ds_val = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
  )

  return ds_train, ds_val

def convert_dataset(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    image = item['image']
    label = item['label']
    image = tf.dtypes.cast(image, 'float32') / 255.
    return image, label