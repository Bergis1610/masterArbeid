import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image as img

BRYGGE_SEKVENS = "./bilder/brygge_sekvens"
BRO_SEKVENS = "./bilder/bro_sekvens"
RESULT_FOLDER = BRYGGE_SEKVENS
save_path = "./bilder/lagra_bilete/"
bilde_navn_disp = "disp_nærme_brygga.png"
bilde_navn_venstre = "nærme_brygga.png"


def main():
    K = np.loadtxt(f"{RESULT_FOLDER}/left/K_matrix.txt")
    R = np.loadtxt(f"{RESULT_FOLDER}/left/R_matrix.txt")
    T = np.loadtxt(f"{RESULT_FOLDER}/left/T_matrix.txt")

    plt.ion()

    # Her løkkes det gjennom alle bildene i mappa
    """ 
    left_images_filenames = list(filter(lambda fn: fn.split(".")[-1]=="png", os.listdir(f"{RESULTS_FOLDER}/left")))
    timestamps = list(map(lambda fn: fn.split(".")[0], left_images_filenames))
    for ti in range(0, len(timestamps)):
        timestamp = timestamps[ti]
        left = cv2.imread(f"{RESULTS_FOLDER}/left/{timestamp}.png")
        right = cv2.imread(f"{RESULTS_FOLDER}/right/{timestamp}.png")
        disp = np.array(cv2.imread(f"{RESULTS_FOLDER}/disp_zed/{timestamp}.png", cv2.IMREAD_ANYDEPTH) / 256.0, dtype=np.float32)
    
        plt.imshow(disp, cmap="turbo")
        plt.colorbar()
        plt.show()
        plt.pause(0.1)
        cv2.imshow("Left image", left)
        cv2.waitKey(100)
        plt.clf()
    cv2.destroyAllWindows()  
    """
    


    # Under åpner vi ti = n'te bildet
    left_images_filenames = list(filter(lambda fn: fn.split(".")[-1]=="png", os.listdir(f"{RESULT_FOLDER}/left")))
    timestamps = list(map(lambda fn: fn.split(".")[0], left_images_filenames))
    ti = len(timestamps)-1
    #ti = 0
    timestamp = timestamps[ti]
    left = cv2.imread(f"{RESULT_FOLDER}/left/{timestamp}.png")
    right = cv2.imread(f"{RESULT_FOLDER}/right/{timestamp}.png")
    disp = np.array(cv2.imread(f"{RESULT_FOLDER}/disp_zed/{timestamp}.png", cv2.IMREAD_ANYDEPTH) / 256.0, dtype=np.float32)

    plt.imshow(disp, cmap="turbo")
    plt.colorbar()
    plt.show()

    lagreDisp = os.path.join(save_path, bilde_navn_disp)
    
    plt.savefig(lagreDisp)

    plt.pause(0.1)
    cv2.imshow("Left image", left)
    cv2.waitKey(0)
    plt.clf()

    lagreVenstre = os.path.join(save_path, bilde_navn_venstre)
    cv2.destroyAllWindows()

    image = img.open(f"{RESULT_FOLDER}/left/{timestamp}.png")
    lagreVenstre = os.path.join(save_path, bilde_navn_venstre)
    image.save(lagreVenstre)

if __name__ == "__main__":
    main()

