from PIL import Image
import numpy as np

# simple image loader (does not 
# account for aspect ratio...yet)
def load_image(path=''):
  image = Image.open(path)
  resized = image.resize((180, 180))
  np_image = np.array(resized)
  np_image = np.expand_dims(np_image, axis=0)
  return np_image