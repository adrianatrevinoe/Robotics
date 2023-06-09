import numpy as np
import matplotlib.pyplot as pyplot
import cv2
from sklearn import linear_model

def region_of_interest(img_name, vertices):
    '''

    '''
    mask = np.zeros_like(img_name)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img_name, mask)
    return masked

def pipeline(img_name):
    '''
    DOC
    '''
    # 1.- Read Image - reduce por 4 la imagen
    img_colour = cv2.imread(img_name,cv2.IMREAD_REDUCED_COLOR_4)

    # Verify that image exists
    if img_colour is None:
        print('Error: image ', img_name, 'could not be read')
        exit()
    #cv2.imshow('Colour image', img_colour)

    # 2.- Convert from BGR to RGV then from RGB to greyscale
    #opencv cuando carga imagen a color lo carga como BGR
    img_colour_rgb = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB) #BGR2GRAY
    grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('Greyscale image', grey)

    # 3.- Apply Gaussian smoothing
    kernel_size = (11,11) #valores impares, mientras + grandes + smooth
    '''
    multiplica cada valor del kernel con el pixel de la imagen, se hace un
    promedio, si es (9,9) tiene 9 casillas el kernel. Ya que se tenga un promedio
    se sobrepone sobre la celda de en medio, y se vuelve a hacer esto con la sig
    fila, despues se baja un fila y asi se va, hace un suavizado entonces
    LA gauseana hace que se tenga un mayor numero en el centro, sigma dice el
    ancho de la distribución
    '''
    blur_grey = cv2.GaussianBlur(grey,kernel_size,sigmaX = 0, sigmaY = 0)
    #cv2.imshow('Smoothed image', blur_grey)

    # 4. Apply Canny edge detector
    low_threshold, high_threshold = 70,100
    edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize = 3)
    #cv2.imshow('Canny image', edges) #imagen de bordes, cambio brusco de
    #instensidad con sus vecinos

    # 5.- get a region of interest using the just created polygon.
    # Define  a region interest Change the below vertices according
    # to input image resolution
    # p1 = (3, 438)
    # p2 = (3,296)
    # p3 = (325,237)
    # p4 = (610,237)
    # p5 = (1010,420)
    # p6 = (1010,420)
    p1, p2, p3, p4, p5, p6 = (3, 438), (3,296), (325,237), (610,237), (1010,420), (1010,420) #vertices de la imagen

    #6. - Filtra todo menos lo que esta en esos vertices
    vertices = np.array([[p1, p2, p3, p4, p5, p6]], dtype = np.int32)
    roi_image = region_of_interest(edges,vertices)
    cv2.imshow("Canny image within Region of interest", roi_image)

    #6.- Apply Hough transform for lane lines detection
    rho = 2                     # distance resolution in px of the Hough grid
    theta = np.pi/100           # distance resolution in px of the Hough grid
    threshold = 20              # min num of votes(intersections in Hough grid)
    min_line_len = 40            # min de pixeles para ser linea
    max_line_gap = 20           # espacio entre lineas para ser 2 o 1
    hough_lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]),
    minLineLength = min_line_len, maxLineGap = max_line_gap)

    # [[x y (de un extremo) x y (del otro extremo)]]
    print(f'Detected lines : \n {hough_lines}')
    print(f'number of lines : \n {hough_lines.shape}')

    # # 7.- Initialise a new image to hold the original image with the detecte dlines
    # img_colour_with_lines= img_colour.copy()
    # img_colour_with_left_and_right_lines = img_colour.copy()
    # img_lane_lines = img_colour.copy()
    # left_lines, left_slope, right_lines, right_slope = list(), list(), list(), list()
    # ymin, ymax, xmin, xmax = 0.0,0.0,0.0,0.0,0.0
    # x_left, y_left, x_right, y_right = list(), list(), list(), list()
    #
    # # Slope and standard deviation for left and right lane lines
    # # This metrics were previosy obtained after analysing the left and right
    # # lane lines for a 50 meter road
    # left_slope_mean, left_slope_std = -20.09187457, 3.40155536 #ro es perpendicular a la linea
    # #left - , right +
    # # compara la pendiente de las lineas de carril con las otras (4)
    # right_slope_mean, right_slope_std = 21.7138409, 1.731189840

    cv2.waitKey(0)

#Test pipeline
img_name = 'G0073201.JPG'
pipeline(img_name)
