
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread(r'18.bmp', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)


edges = cv2.Canny(blurred_image, 3,5)


plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.imwrite(r'canny18.png',edges)