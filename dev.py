from skimage.filters import try_all_threshold
from skimage import data
import matplotlib.pyplot as plt
import cv2
# img = data.page()
# print(img)
img = cv2.imread('images.jpeg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Here, we specify a radius for local thresholding algorithms.
# If it is not specified, only global algorithms are called.
fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()