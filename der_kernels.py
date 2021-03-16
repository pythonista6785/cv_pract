# >>> import cv2
# >>> cv2.__version__
# '4.5.1'
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("img/roman_art.jpeg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, figsize=(12, 8))
plt.imshow(image)