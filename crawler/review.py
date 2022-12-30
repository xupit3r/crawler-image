import matplotlib.pyplot as plt
import numpy as np

# plot the result of a training run
def plot_result(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, "bo", label="Training loss")
  plt.plot(epochs, val_loss, "b", label="Validation loss")
  plt.title("Training and validation loss")
  plt.legend()
  plt.show()


def review_dataset(dataset):
  class_names = dataset.class_names
  plt.figure(figsize=(10, 10))
  for images, labels in dataset.take(1):
    for i in range(9):
      plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[np.argmax(labels[i])])
      plt.axis("off")
  plt.show()

