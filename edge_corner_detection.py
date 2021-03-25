import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load a color image in grscale

col_img = cv2.imread("img/roman_art.jpeg", 1)  # 0 for grayscale, 1 for color image
# cv2.imshow("image", col_img)

# Canny Edge Detection
img = cv2.imread("img/roman_art.jpeg", 0)
edges = cv2.Canny(img, 50, 200)

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap="gray")
plt.title("Edge Image"), plt.xticks([]), plt.yticks([])
plt.show()

# Harris Corner Detection
corn_img = cv2.imread("img/chess_board.png")
gray = cv2.cvtColor(corn_img, cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image
corn_img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow("dst", corn_img)

cv2.waitKey(0)
cv2.destroyAllWindows()