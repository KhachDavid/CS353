# This file has the functions that you need to implement. 
import numpy as np
import scipy as sp
import skimage as ski
import matplotlib.pyplot as plt

# Function 1:
def loadCameraman(imsize: tuple) -> np.ndarray:
    """
    This function loads the cameraman image and returns a resized version of it. 
    The cameraman image is a standard test image widely used in the field of image processing. 
    The image is grayscale and has a size of 256x256 pixels. 
    The function should return a resized version of the cameraman image with the size of (imsize[0] x imsize[1]) pixels. 
    The resizing should be done using the cubic interpolation. 
    The function should return the resized image as a numpy array.
    camera.png is provided to you in the assignment folder. You can hardcode its location in the function.
    After cubic interpolation, the pixel values should be clipped to the range [0, 1].
    input:
    - imsize: a tuple of two integers (height, width) specifying the size of the resized image
    output:
    - img: a numpy array of size (imsize[0] x imsize[1]) containing the resized cameraman image
    Useful functions:
    plt.imread: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
    ski.transform.resize: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
    np.clip: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    """
    # Write your code here
    img = None
    # end of your code
    
    return img

# Function 2:
def displayGray(img: np.ndarray) -> None:
    """
    This function displays a grayscale image. 
    The function should display the input image using the grayscale colormap. 
    The function should not return anything. 
    input:
    - img: a numpy array of size (height x width) containing a grayscale image
    output:
    - None
    Useful functions:
    plt.imshow: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    plt.colorbar: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
    Look into 'cmap' argument of plt.imshow.
    Don't forget to add a color bar!
    There are no axes labels in this function as this function has a general purpose.
    """
    plt.figure()
    # Write your code here

    # end of your code
    plt.show()
    return

# Function 3:
def generate2DGaussianKernel(kernel_size:tuple, std:float) -> np.ndarray:
    """
    - This function generates a 2D Gaussian kernel. 
    - The function should return a 2D Gaussian kernel of size (kernel_size[0] x kernel_size[1]).
    - Our kernel is isotropic, meaning that the standard deviation is the same in both the x and y directions.
    - Our kernel's center is at orign (0,0). KERNEL DIMENSIONS ARE ALWAYS ODD!
    - The kernel should be normalized such that the sum of all its elements is equal to 1. 
    - The function should return the kernel as a numpy array. 
    - You are NOT allowed to use any built-in functions that generate the 2D Gaussian kernel for you (i.e 
      scipy.ndimage.gaussian_filter, cv2.getGaussianKernel, etc.)
    **input:
    - kernel_size: a tuple of two integers (height, width) specifying the size of the kernel. Both height and width 
    are ODD numbers!
    - std: a float specifying the standard deviation of the Gaussian kernel
    output:
    - kernel: a numpy array of size (kernel_size[0] x kernel_size[1]) containing the 2D Gaussian kernel
    Useful functions:
    np.arange: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    np.meshgrid: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    np.exp: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
    """
    # Write your code here, This is a guided example:

    kernel = None

    # end of your code
    return kernel

# Function 4:
def spatial2DConvolution(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    """
    This function performs 2D spatial convolution between an image and a kernel. 
    The function should return the result of the convolution. 
    The function should use zero-padding to handle the borders of the image. (It shold do a full convolution.)
    The function should return the result as a numpy array.
    Do not use fourier transforms to perform the convolution.
    input:
    - image: a numpy array of size (height x width) containing a grayscale image
    - kernel: a numpy array of size (kernel_height x kernel_width) containing a 2D kernel
    output:
    - result: a numpy array of size (height + kernel_height - 1 x width + kernel_width -1) 
    containing the result of the convolution
    Useful functions:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    Read the mode argument of the function carefully.
    """
    # Write your code here

    result = None
    # end of your code
    
    return result

# Function 5:
def brightnessHist(img:np.ndarray, range:tuple = (0,1), num_bins:int = 25, autograder_call:bool = False) -> None:
    """
    - This function plots the histogram of the brightness values of a grayscale image. 
    - The function should display the histogram of the input image. 
    - The histogram should be computed using num_bins bins and the range of the histogram should be specified by the
      range argument. 
    - The function should not return anything. 
    input:
    - img: a numpy array of size (height x width) containing a grayscale image
    - range: a tuple of two floats (min, max) specifying the range of the histogram
    - num_bins: an integer specifying the number of bins in the histogram
    - autograder_call: a boolean that is set to True only when the function is called from the autograder.
    output:
    - You are expected to return the boundaries and count of each bin in the histogram. np.hist handles this for you.
    Useful functions:
    np.hist: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    plt.bar: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
    Don't forget to add axes labels and a title to the plot!
    """
    plt.figure()
    # Write your code here
    histogramCounts, binEdges = None, None
    # end of your code
    if autograder_call:
        plt.close()
        return histogramCounts, binEdges
    plt.show()
    return
