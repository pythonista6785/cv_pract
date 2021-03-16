from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


# construct a identity filter
identity = np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype="int")

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))

largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")

# construct a Low Pass Filter (lpf)
lpf = np.ones((3, 3), dtype="float") * (1.0 / 9)

# construct a High Pass Filter (hpf)
hpf = np.array(([-1, -1, -1], [1, 8, -1], [-1, -1, -1]), dtype="float") * (1 / 9)

# construct a gaussian blur 3 X 3
GaussianBlur = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1]), dtype="float") * (1 / 16)

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype="int")

# construct the argumnet parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())


# construct the kernel bank, a list of kernels we're going
# to apply OpenCV's 'filter2D' function

kernelBank = (
    ("identity", identity),
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("low_pass_filter", lpf),
    ("high_pass_filter", hpf),
    ("gaussian_3by3", GaussianBlur),
    ("laplacian", laplacian),
    ("sobelX", sobelX),
    ("sobelY", sobelY),
)

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, kernel) in kernelBank:
    # apply the kernel to the grayscale image using
    # openCV's 'filter2D' function
    print("[INFO] applying {} kernel".format(kernelName))
    opencvOutput = cv2.filter2D(gray, -1, kernel)

    # show the output images
    cv2.imshow("Original", gray)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
