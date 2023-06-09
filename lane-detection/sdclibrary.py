import numpy as np
import matplotlib.pyplot as pyplot
import cv2
from sklearn import linear_model
import statistics as stat

def read_image(img_name):
    '''
    reduces an image by 4, if there is no img an error show up
    '''
    img_colour = cv2.imread(img_name,cv2.IMREAD_REDUCED_COLOR_4)
    if img_colour is None:
        print('Error: image ', img_name, 'could not be read')
        exit()
    return img_colour

def BGR_to_GRAY(colour_img):
    '''
    converts an img from bgr color to gray
    '''
    img_colour_rgb = cv2.cvtColor(colour_img, cv2.COLOR_BGR2RGB) #BGR2GRAY
    grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)
    return grey

def gaussian_smoothing(grey_img, kernel):
    '''
    multiplica cada valor del kernel con el pixel de la imagen, se hace un
    promedio, si es (9,9) tiene 9 casillas el kernel. Ya que se tenga un promedio
    se sobrepone sobre la celda de en medio, y se vuelve a hacer esto con la sig
    fila, despues se baja un fila y asi se va, hace un suavizado entonces
    LA gauseana hace que se tenga un mayor numero en el centro, sigma dice el
    ancho de la distribución
    '''
    kernel_size = (kernel,kernel) #valores impares, mientras + grandes + smooth
    blur_grey = cv2.GaussianBlur(grey_img,kernel_size,sigmaX = 0, sigmaY = 0)
    return blur_grey

def canny_edge(blurred_img, low_threshold, high_threshold, aperture_size):
    '''
    imagen de bordes, cambio brusco de instensidad con sus vecinos
    '''
    edges = cv2.Canny(blurred_img, low_threshold, high_threshold, apertureSize = aperture_size)
    #imagen de bordes, cambio brusco de
    #instensidad con sus vecinos
    return edges

def region_of_interest(colour_img, vertices):
    '''
    creates a mask where we fill the image with 0's
    then we create a mask where the roi will have the pixels at 255
    (filling it with white and to the max) the region is marked with vertices,
    the merging is done with the 'and'.
    '''
    mask = np.zeros_like(colour_img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(colour_img, mask)
    return masked

def hough(roi_image, rho, theta, threshold, min_line_len,max_line_gap):
    '''
        returns hough lines

        rho:  distance resolution in px of the Hough grid
        theta: distance resolution in px of the Hough grid
        threshold: min num of votes(intersections in Hough grid)
        min_line_len: min de pixeles para ser linea
        max_line_gap: espacio entre lineas para ser 2 o 1
    '''
    hough_lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]),
    minLineLength = min_line_len, maxLineGap = max_line_gap)
    return hough_lines

def merge_lines(x,y):
    '''
    two arrays are given and we get the min and max from x and y.
    '''
    ymin = np.min(y)
    ymax = np.max(y)
    xmin = np.min(x)
    xmax = np.max(x)
    return xmin, xmax, ymin, ymax

def roi_between_lines(img, xminL,xmaxL,yminL,ymaxL,xminR,xmaxR,yminR,ymaxR):
    '''
    gives region of interest between both detected lines, the min and max are
    passed and put into vertices to get the highest and lowest four coords (x,y)
    from left and right.

    '''
    v1, v2, v3, v4 = (xminL,ymaxL), (xmaxR,ymaxR),(xminR,yminR), (xmaxL,yminL)
    vertices_lines = np.array([[v1,v2,v3,v4]], dtype = np.int32)
    mask_lines = np.zeros_like(img)
    cv2.fillPoly(mask_lines, vertices_lines, [241,255,1])
    region = cv2.bitwise_and(img, mask_lines)

    area_image = cv2.addWeighted(img,1,region,1,0)

    return area_image


def pipeline(image_name):
    '''+
    Function to detect the lanes on a road, we have a process of 9 steps
    to get to the final image where we have the lines detected with its region of
    interest. This image will show 7 images.
    A color image, gray image, blurred gray image, canny img, image with region of interest,
    lines of hough detected, lines of hough detected with different colors for left and right,
    and the lines with a region of interest in the middle.
    '''
    # 1.- Read Image - reduce por 4 la imagen
    img_colour = read_image(image_name)
    cv2.imshow('Colour image', img_colour)

    # 2.- Convert from BGR to RGV then from RGB to greyscale
    grey = BGR_to_GRAY(img_colour)
    cv2.imshow('Greyscale image', grey)

    # 3.- Apply Gaussian smoothing
    blur_grey = gaussian_smoothing(grey, 13)
    cv2.imshow('Smoothed image', blur_grey)

    # 4. Apply Canny edge detector
    edges = canny_edge(blur_grey,70,100,3)
    cv2.imshow('Canny image', edges)

    # 5.- get a region of interest using the just created polygon.
    # Define  a region interest Change the below vertices according
    # to input image resolution
    p1, p2, p3, p4, p5, p6 = (1, 420), (1,300), (300,220), (585,220), (1010,380), (1010,420)
    #vertices de la imagen

    #6. - Filtra todo menos lo que esta en esos vertices
    vertices = np.array([[p1, p2, p3, p4, p5, p6]], dtype = np.int32)
    roi_image = region_of_interest(edges,vertices)
    cv2.imshow("Canny image within Region of interest", roi_image)

    #6.- Apply Hough transform for lane lines detection
    hough_lines = hough(roi_image,2, np.pi/100, 16, 45,50)
    #[[x y (de un extremo) x y (del otro extremo)]]
    #print(f'Detected lines : \n {hough_lines}')
    #print(f'number of lines : \n {hough_lines.shape}')

    # 7.- Initialise a new image to hold the original image with the detected lines
    img_colour_with_lines= img_colour.copy()
    img_colour_with_left_and_right_lines = img_colour.copy()
    img_lane_lines = img_colour.copy()

    left_lines, left_slope, right_lines, right_slope, other = list(), list(), list(), list(), list()
    ymin, ymax, xmin, xmax = 0.0, 0.0, 0.0, 0.0
    x_left, y_left, x_right, y_right = list(), list(), list(), list()

    # Slope and standard deviation for left and right lane lines
    # This metrics were previosy obtained after analysing the left and right
    # lane lines for a 50 meter road
    left_slope_mean, left_slope_std = -20.09187457, 3.40155536 #ro es perpendicular a la linea
    #left - , right +
    # compara la pendiente de las lineas de carril con las otras (4)
    right_slope_mean, right_slope_std = 21.7138409, 1.731189840

    for line in hough_lines:
        for x1, y1, x2, y2 in line:

            # find slope
            slope = (y2-y1)/(x2-x1)
            if (x1 or x2) > 400:
                if (slope > 0.33) and (slope < 0.5): #positive slope - right lanr
                    right_slope.append(slope)
                    right_lines.append(line)

                    x_right.append(x1)
                    x_right.append(x2)
                    y_right.append(y1)
                    y_right.append(y2)

                    cv2.line(img_colour_with_lines, (x1,y1), (x2,y2), (160,0,255), 3) #left
                    cv2.line(img_colour_with_left_and_right_lines, (x1,y1), (x2,y2), (0,0,255), 3) #color

            #elif if (x1 or x2) < 400:
            elif (slope < -0.315) and (slope > -0.55): #negative slope - left lane
                x_left.append(x1)
                x_left.append(x2)
                y_left.append(y1)
                y_left.append(y2)

                left_slope.append(slope)
                left_lines.append(line)

                cv2.line(img_colour_with_lines, (x1,y1), (x2,y2), (160,0,255), 3) #right
                cv2.line(img_colour_with_left_and_right_lines, (x1,y1), (x2,y2), (255,0,0), 3) #color

            #if the slope doesn't match lane requirements don't append it
            else:
                #just to verify they filtered correctly
                other.append(slope)
    #draw lines
    cv2.imshow("Detected Hough transofrm lines", img_colour_with_lines) #initial
    cv2.imshow("Inlier left and right Hough lines", img_colour_with_left_and_right_lines) #red and blue

    # 8. Merge both lines from left and both lines from y
    xmin_left, xmax_left, ymin_left, ymax_left = merge_lines(x_left,y_left)
    xmin_right, xmax_right, ymin_right, ymax_right = merge_lines(x_right,y_right)
    cv2.line(img_lane_lines, (xmax_left,ymin_left), (xmin_left,ymax_left), (255,255,0), 3) #left
    cv2.line(img_lane_lines, (xmax_right, ymax_right), (xmin_right, ymin_right), (255,255,0), 3) #right

    #9. - Create region of interest between two lines
    img_lane_lines_area = roi_between_lines(img_lane_lines,xmin_left,xmax_left,ymin_left,ymax_left,xmin_right,xmax_right,ymin_right,ymax_right)
    cv2.imshow("Left and right lane lines - region", img_lane_lines_area) #merge lines and color space between

    cv2.waitKey(0)
