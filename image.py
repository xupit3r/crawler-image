from crawler.datasets import get_fashion_mnist, get_food101, get_flowers, get_svhn, get_domainnet
from crawler.train import train

options = {
  'show_summary': False,
  'save_model': False,
  'show_examples': False,
  'early_stop': True,
  'plot_results': False,
  'batch_size': 32,
  'image_height': 32,
  'image_width': 32,
  'channels': 1
}

train(
  options=options,
  dataset=get_domainnet(
    image_height=options['image_height'],
    image_width=options['image_width']
  ),
  model_name='image_joe'
)