import tensorflow as tf
import tensorflow_datasets as tfds


def convert_data(image, label, num_classes):
  return (tf.cast(image, tf.float32) / 255., tf.one_hot(label, num_classes))

def convert_data_resize(images, labels, num_classes, image_height, image_width):
  images = tf.cast(images, tf.float32) / 255
  images = tf.image.resize(images, (image_height, image_width))
  labels = tf.one_hot(labels, num_classes)
  return (images, labels)

def get_image_dataset(name='', split=[], dimensions=[], converter=convert_data):
  [train, val], info = tfds.load(
    name,
    split=split, 
    shuffle_files=True,
    as_supervised=True,
    with_info=True
  )

  num_classes = info.features['label'].num_classes
  ds_train = train.map(lambda image, label: converter(image, label, num_classes, dimensions[0], dimensions[1]))
  ds_val = val.map(lambda image, label: converter(image, label, num_classes, dimensions[0], dimensions[1]))

  return {
    'ds_train': ds_train,
    'ds_val': ds_val,
    'num_classes': num_classes
  }

def get_fashion_mnist():
  return get_image_dataset(
    'fashion_mnist',
    ['train', 'test']
  )

def get_food101(image_height=28, image_width=28):
  return get_image_dataset(
    'food101',
    ['train', 'validation'],
    [image_height, image_width],
    convert_data_resize
  )

def get_flowers(image_height=28, image_width=28):
  return get_image_dataset(
    'tf_flower',
    ['train[80%:]', 'train[:20%]'],
    [image_height, image_width],
    convert_data_resize
  )

def get_svhn(image_height=32, image_width=32):
  return get_image_dataset(
    'svhn_cropped',
    ['train', 'test'],
    [image_height, image_width],
    convert_data_resize
  )

def get_domainnet(image_height=32, image_width=32):
  return get_image_dataset(
    'domainnet',
    ['train', 'test'],
    [image_height, image_width],
    convert_data_resize
  )