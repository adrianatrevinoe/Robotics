import argparse
from sdclibrary import pipeline
import os
import time
import cv2

# #Test pipeline
# image_name = './sdc-dataset-adv-rob/G0073354.JPG'
# pipeline(image_name)

# Ask the user to enter the path to input images
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_images", help ="Path to input images")
args = parser.parse_args()

# Get the list of image files and sort it alphabetically
list_with_name_of_images = sorted(os.listdir(args.path_to_images))

# Loop through each input image
for im in list_with_name_of_images:

    # Build path and image name
    path_and_im = args.path_to_images +im

    # Get the start time
    start_time = time.process_time()

    # Run the workflow to each input image
    try:
        pipeline(path_and_im)
    except:
        print("Error, image can't be read")
    # Print the name of image being processed and compute FPS
    print ( f"Processing image:{path_and_im} ", f"\tCPU execution time:{1/(time.process_time()-start_time):0.4f} FPS" )

    # If the user presses the key ' q ' ,
    # the program finishes
    if cv2 .waitKey( 1 ) & 0xFF ==ord ('q'):
        print ( "\nProgram interrupted by the user - bye mate!" )
        break
