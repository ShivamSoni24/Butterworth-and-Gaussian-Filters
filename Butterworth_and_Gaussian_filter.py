#importing necessary libraries and functions for images and mathematical functions
import numpy as np                  #numpy library for mathematical functions
import matplotlib.pyplot as plt     #matplotlib for plotting of the image
from PIL import Image, ImageOps     #Pillow library for openinng image
from scipy.fft import fft2, ifft2   #fourier transform and inverse fourier transform for 2D array

#For ignoring divide by zero error
np.seterr(divide='ignore', invalid='ignore')

#function to plot 2 images side be side in 2 rows
def plot(data, title, n=2):
    plot.i += 1
    plt.xticks([])
    plt.yticks([])
    plt.subplot(n,2,plot.i)
    plt.imshow(data)
    if (plot.i == 1):
        pass
    else:
        plt.gray()
    plt.title(title)

#variable for plotting images
plot.i = 0

#function for creating the filter with image dimensions
def creating_filter(img_path="images\\flash1.jpg"):
    
    #opening the image with full path of image
    global input_image
    input_image = Image.open(img_path)
    #converting the image from colored to grayscale. It converts 3D to 2D
    input_image_gs = ImageOps.grayscale(input_image)

    #Printing Image format, dimension and mode in which it is opened.
    """print(input_image.format)
    print(input_image_gs.size)
    print(input_image_gs.mode)"""

    #plotting image using pyplot 
    """plt.imshow(input_image_gs)
    plt.gray()"""
    #plt.show()

    #converting image to numpy array and finding its dimension
    input_image_gs_array = np.asarray(input_image_gs)
    dimensions = input_image_gs_array.shape

    #Copying the image dimensions for the filter
    filter_height = dimensions[0]
    filter_width = dimensions[1]

    #finding fourier transform of the image using fft2 function of scipy
    fourier_transform = fft2(input_image_gs)
    #round_function = lambda x: round(x,2)
    #fourier_transform_rounded = [list(map(round_function, i)) for i in fourier_transform]
    #print(fourier_transform)

    #Making fix order
    #n = order
    #Cut-off frequency value
    #D0 = cutoff

    #Designing the filter
    u = np.arange(0,filter_height)
    v = np.arange(0,filter_width)
    #print(u,v)

    #converting u array with values (filter_height/2, -filter_height/2)
    idx = []
    for i in u:
        if(i> (filter_height/2)):
            idx.append(i)
    #print(u[idx])
    #print(idx)
    u[idx] = (u[idx] - filter_height)
    #print(u)

    #converting v array with values (filter_width/2, -filter_width/2)
    idy = []
    for i in v:
        if(i > (filter_width/2)):
            idy.append(i)
    v[idy] = (v[idy] - filter_width)

    #print(v[idy])
    #print(idy)
    #print(v)
    #converting list to numpy array
    u = np.array(u)
    v = np.array(v)
    #print(u,v)

    #meshgrid is function that returns coordinates of v and u. List V with each row is a copy of v and list U with each column is a copy of u 
    U, V = np.meshgrid(v,u)
    #print(U.shape, V.shape)
    #print(U)
    #returning the filter and original image array
    return U, V, input_image_gs_array, fourier_transform

#function that performs butterworth highpass function
def butterworth_highpass_function(img_path="images\\flash1.jpg",order=3,cutoff=10):

    #creating filter for the function
    U, V, input_image_gs_array, fourier_transform = creating_filter(img_path=img_path)

    #Making fix order
    n = order
    #Cut-off frequency value
    D0 = cutoff

    #Euclidean distance
    D = np.sqrt(U**2 + V**2)

    #Determining filtering mask
    H = (1/(1 + (D0/D)**(2*n)))

    #Convolution between the Fourier Transformed image and the mask
    G = H * fourier_transform

    #finding inverse fourier transform of the image using ifft2 function of scipy
    output_image_array = ifft2(G)
    #separating real part from complex numbers
    output_image_array_real = output_image_array.real
    #print(output_image_array_real.shape)

    #plotting original image and transformed image
    n=str(n)
    D0=str(D0)
    name = str("BHPF with n="+n+" Cut-off frequency="+D0)
    """plot(input_image_gs_array,"Orignal Image")
    plot(output_image_array_real,name)
    plt.show()"""

    return input_image, output_image_array_real, name

#function that performs butterworth lowpass function
def butterworth_lowpass_function(img_path="images\\flash1.jpg",order=3,cutoff=10):

    #creating filter for the function
    U, V, input_image_gs_array, fourier_transform = creating_filter(img_path=img_path)

    #Making fix order
    n = order
    #Cut-off frequency value
    D0 = cutoff

    #Euclidean distance
    D = np.sqrt((U**2 + V**2))

    #Determining filtering mask
    H = (1/(1 + (D/D0)**(2*n)))
    #converting high pass to low pass
    #HL=1-H

    #Convolution between the Fourier Transformed image and the mask
    G = H * fourier_transform

    #finding inverse fourier transform of the image using ifft2 function of scipy
    output_image_array = ifft2(G)
    #separating real part from complex numbers
    output_image_array_real = output_image_array.real
    #print(output_image_array_real.shape)

    #plotting original image and transformed image
    n=str(n)
    D0=str(D0)
    name = str("BLPF with n="+n+" Cut-off frequency="+D0)
    """plot(input_image_gs_array,"Orignal Image")
    plot(output_image_array_real,name)
    plt.show()"""

    return input_image_gs_array, output_image_array_real, name

#main function which will convert and give output image as converted to highpass and lowpass butterworth filter
def butterworth_HP_LP_filter(img_path="images\\flash1.jpg",order=3,cutoff=10):
    
    #give image path using keyword img_path=, order by order= (by default 3), cutoff fq by cutoff= (by default 10) for both HP and LP
    org, output, title = butterworth_highpass_function(img_path=img_path, order=order, cutoff=cutoff)
    plot(org,"Original")
    plot(output,title)

    #give image path using keyword img_path=, order by order= (by default 3), cutoff fq by cutoff= (by default 10) for both HP and LP
    org, output, title = butterworth_lowpass_function(img_path=img_path, order=order, cutoff=cutoff)
    plot(org,"Grayscale")
    plot(output,title)

    #show the image
    plt.show()

#function to perform lowpass gaussian function
def gaussian_lowpass_function(img_path="images\\flash1.jpg",cutoff=10):
    
    #creating filter for the function
    U, V, input_image_gs_array, fourier_transform = creating_filter(img_path=img_path)
    
    #Cut-off frequency value
    D0 = cutoff

    #Euclidean distance
    D = (U**2 + V**2)
    D = -(D/(2*(D0**2)))
    H = np.exp(D)

    #Convolution between the Fourier Transformed image and the mask
    G = H * fourier_transform

    #finding inverse fourier transform of the image using ifft2 function of scipy
    output_image_array = ifft2(G)
    #separating real part from complex numbers
    output_image_array_real = output_image_array.real
    #print(output_image_array_real.shape)


    #plotting original image and transformed image
    D0=str(D0)
    name = str("Gaussian Lowpass Filter with Cut-off frequency="+D0)
    """plot(input_image_gs_array,"Orignal Image")
    plot(output_image_array_real,name)
    plt.show()"""
    return input_image, output_image_array_real, name

#function to peerform gaussian highpass filter
def gaussian_highpass_function(img_path="images\\flash1.jpg",cutoff=10):
    
    #creating filter for the function
    U, V, input_image_gs_array, fourier_transform = creating_filter(img_path=img_path)
    
    #Cut-off frequency value
    D0 = cutoff

    #Euclidean distance
    D = (U**2 + V**2)
    D = -(D/(2*(D0**2)))
    H = np.exp(D)
    #converting lowpass to highpass
    HP = 1 - H

    #Convolution between the Fourier Transformed image and the mask
    G = HP * fourier_transform

    #finding inverse fourier transform of the image using ifft2 function of scipy
    output_image_array = ifft2(G)
    #separating real part from complex numbers
    output_image_array_real = output_image_array.real
    #print(output_image_array_real.shape)

    #plotting original image and transformed image
    D0=str(D0)
    name = str("Gaussian Highpass Filter with Cut-off frequency="+D0)
    """plot(input_image_gs_array,"Orignal Image")
    plot(output_image_array_real,name)
    plt.show()"""
    return input_image_gs_array, output_image_array_real, name

#main function which will convert and give output image as converted to highpass and lowpass butterworth filter
def gaussian_LP_HP_filter(img_path="images\\flash1.jpg",cutoff=10):
    
    #give image path using keyword img_path=, order by order= (by default 3), cutoff fq by cutoff= (by default 10) for both HP and LP
    org, output, title = gaussian_lowpass_function(img_path=img_path, cutoff=cutoff)
    plot(org,"Original")
    plot(output,title)

    #give image path using keyword img_path=, order by order= (by default 3), cutoff fq by cutoff= (by default 10) for both HP and LP
    org, output, title = gaussian_highpass_function(img_path=img_path, cutoff=cutoff)
    plot(org,"Grayscale")
    plot(output,title)

    #show the image
    plt.show()

#function to plot all images in one frame, both butterworth and gaussian filter
def plot_together(img_path="images\\flash1.jpg",order=3,cutoff=10):
    
    #give image path using keyword img_path=, order by order= (by default 3), cutoff fq by cutoff= (by default 10) for both HP and LP
    org, output, title = butterworth_highpass_function(img_path=img_path, order=5, cutoff=20)
    #plotting image with colour
    plot(org,"Original",n=3)
    
    org1, output1, title1 = butterworth_lowpass_function(img_path=img_path, order=10, cutoff=cutoff)
    #plotting grayscale image
    plot(org1,"Grayscale",n=3)
    #plotting butterworth lowpass
    plot(output1,title1,n=3)
    #plotting butterworth highpass
    plot(output,title,n=3)

    #give image path using keyword img_path=, order by order= (by default 3), cutoff fq by cutoff= (by default 10) for both HP and LP
    org, output, title = gaussian_lowpass_function(img_path=img_path, cutoff=8)
    #plotting gaussian lowpass
    plot(output,title,n=3)

    #give image path using keyword img_path=, order by order= (by default 3), cutoff fq by cutoff= (by default 10) for both HP and LP
    org, output, title = gaussian_highpass_function(img_path=img_path, cutoff=5)
    #plotting gaussian highpass
    plot(output,title,n=3)

    #show the image
    plt.show()

def plot_separate():
    #executing main function
    #displaying output of butterworth filter
    butterworth_HP_LP_filter(img_path="images\\Batman3.jpg",order=3,cutoff=25)

    #changing value of plot.i = 0 for printing new image
    plot.i=0

    #displaying output of gaussian filter
    gaussian_LP_HP_filter(img_path="images\\Batman3.jpg",cutoff=35)

#main function to plot and compare
#function to first separately show output
plot_separate()
plot.i=0
#function to plot and compare together
plot_together()