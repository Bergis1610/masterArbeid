import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image as img

BRYGGE_SEKVENS = "./bilder/brygge_sekvens"
BRO_SEKVENS = "./bilder/bro_sekvens"
LAGRA_BILDER = ".bilder/lagra_bilete"
RESULT_FOLDER = BRYGGE_SEKVENS

def main():
    K = np.loadtxt(f"{RESULT_FOLDER}/left/K_matrix.txt")
    R = np.loadtxt(f"{RESULT_FOLDER}/left/R_matrix.txt")
    T = np.loadtxt(f"{RESULT_FOLDER}/left/T_matrix.txt")

    plt.ion()

    # Under åpner vi ti = n'te bildet
    left_images_filenames = list(filter(lambda fn: fn.split(".")[-1]=="png", os.listdir(f"{RESULT_FOLDER}/left")))
    timestamps = list(map(lambda fn: fn.split(".")[0], left_images_filenames))
    ti = 0
    timestamp = timestamps[ti]
    left = cv2.imread(f"{RESULT_FOLDER}/left/{timestamp}.png")
    right = cv2.imread(f"{RESULT_FOLDER}/right/{timestamp}.png")
    disp = np.array(cv2.imread(f"{RESULT_FOLDER}/disp_zed/{timestamp}.png", cv2.IMREAD_ANYDEPTH) / 256.0, dtype=np.float32)


    


    
    plt.imshow(disp, cmap="turbo")
    plt.colorbar()
    plt.show()
 
    cv2.destroyAllWindows() 
    


    # Forsøk på å plotte en kolonne i disp bildet 
    #disparity_map = cv2.imread(f"{LAGRA_BILDER}/disp_første.png", cv2.IMREAD_GRAYSCALE)
    disparity_map = disp

    column_index = disparity_map.shape[1] // 2

    column_data = disparity_map[:, column_index]

    y_values = range(len(column_data))

    plt.figure()
    plt.plot(column_data, y_values)
    plt.gca().invert_yaxis()  # Invert y-axis to match image orientation
    plt.xlabel('Disparity Value')
    plt.ylabel('Image Y-Axis')
    plt.title('Disparity Values Along a Column')
    plt.show()
    plt.waitforbuttonpress()



if __name__ == "__main__":
    main()





""" 
disparity_map = cv2.imread('path/to/your/disparity_map.png', cv2.IMREAD_GRAYSCALE)

# Choose a column index (for example, the middle column)
column_index = disparity_map.shape[1] // 2

# Extract the column
column_data = disparity_map[:, column_index]

# Create the y-axis values corresponding to each pixel row
y_values = range(len(column_data))

# Plotting
plt.figure()
plt.plot(column_data, y_values)
plt.gca().invert_yaxis()  # Invert y-axis to match image orientation
plt.xlabel('Disparity Value')
plt.ylabel('Image Y-Axis')
plt.title('Disparity Values Along a Column')
plt.show()
"""