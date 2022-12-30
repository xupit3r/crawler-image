import numpy as np
from crawler.models import pizza_predict
from crawler.utils import load_image

notPizzaImg = load_image('pictures/hot-dog.jpg')
pizzaImg = load_image('pictures/pizza.jpg')
print(pizza_predict(notPizzaImg))
print(pizza_predict(pizzaImg))