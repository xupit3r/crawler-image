import tensorflow as tf
import pathlib
import tensorflow_datasets as tfds


def convert_data(image, label, num_classes):
  return (tf.cast(image, tf.float32) / 255., tf.one_hot(label, num_classes))

def convert_data_resize(images, labels, num_classes, image_height, image_width):
  images = tf.cast(images, tf.float32) / 255
  images = tf.image.resize(images, (image_height, image_width))
  labels = tf.one_hot(labels, num_classes)
  return (images, labels)
  

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
    image_size=(image_height, image_width)
  )
  

  ds_val = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(image_height, image_width)
  )

  return ds_train, ds_val

def get_fashion_mnist():
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

  return {
    'ds_train': ds_train,
    'ds_val': ds_val,
    'num_classes': num_classes
  }

def get_food101(image_height=28, image_width=28):
  [train, val], info = tfds.load(
    'food101',
    split=['train', 'validation'], 
    shuffle_files=True,
    as_supervised=True,
    with_info=True
  )

  num_classes = info.features['label'].num_classes
  ds_train = train.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))
  ds_val = val.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))

  return {
    'ds_train': ds_train,
    'ds_val': ds_val,
    'num_classes': num_classes
  }

def get_flowers(image_height=28, image_width=28):
  [train, val], info = tfds.load(
    'tf_flowers',
    split=['train[80%:]', 'train[:20%]'], 
    shuffle_files=True,
    as_supervised=True,
    with_info=True
  )

  num_classes = info.features['label'].num_classes
  ds_train = train.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))
  ds_val = val.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))

  return {
    'ds_train': ds_train,
    'ds_val': ds_val,
    'num_classes': num_classes
  }

def get_svhn(image_height=32, image_width=32):
  [train, val], info = tfds.load(
    'svhn_cropped',
    split=['train', 'test'], 
    shuffle_files=True,
    as_supervised=True,
    with_info=True
  )

  num_classes = info.features['label'].num_classes
  ds_train = train.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))
  ds_val = val.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))

  return {
    'ds_train': ds_train,
    'ds_val': ds_val,
    'num_classes': num_classes
  }

def get_domainnet(image_height=32, image_width=32):
  [train, val], info = tfds.load(
    'domainnet',
    split=['train', 'test'], 
    shuffle_files=True,
    as_supervised=True,
    with_info=True
  )

  num_classes = info.features['label'].num_classes
  ds_train = train.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))
  ds_val = val.map(lambda image, label: convert_data_resize(image, label, num_classes, image_height, image_width))

  return {
    'ds_train': ds_train,
    'ds_val': ds_val,
    'num_classes': num_classes
  }