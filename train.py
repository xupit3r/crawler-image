import numpy as np
from crawler.image import train

train(
  show_summary=True,
  save_model=True,
  early_stop=True,
  show_examples=False,
  plot_results=True
)