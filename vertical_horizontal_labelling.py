import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image as img

from sklearn.cluster import KMeans

BRYGGE_SEKVENS = "./bilder/brygge_sekvens"
BRO_SEKVENS = "./bilder/bro_sekvens"
LAGRA_BILDER = ".bilder/lagra_bilete"
RESULT_FOLDER = BRYGGE_SEKVENS

def interpolate_column(column):
    # Indices of valid and invalid elements
    # Adds the indices of values that are valid meaning real numbers to valid_indices and invalid indices meaning inf, -inf and NaN to 
    valid_mask = np.isfinite(column)
    valid_indices = np.where(valid_mask)[0]
    invalid_indices = np.where(~valid_mask)[0]

    # Check if we have enough data for interpolation
    if len(valid_indices) == 0:
        # No valid data in this column
        print("No valid data")
        return column
    """ elif len(invalid_indices) == 0:
        # No need for interpolation
        print("Not Necessary")
        return column """

    #count +=1 
    # Interpolate invalid data points
    valid_data = column[valid_mask]
    column[~valid_mask] = np.interp(invalid_indices, valid_indices, valid_data)
    return column

def interpolate_each_column(dI):
    # Applying the interpolation to each column
    height, width = dI.shape
    for x in range(width):
        dI[:, x] = interpolate_column(dI[:, x])

    return dI

n = 5  # Size of the kernel, as Vipul used in his thesis
kernerl_k = np.ones(n) / n  # Kernel for averaging
def low_pass_filter(column, K = kernerl_k):
    n = 5  # Size of the kernel, as Vipul used in his thesis
    kernerl_k = np.ones(n) / n  # Kernel for averaging
    K = kernerl_k
    # Apply convolution
    filtered_column = np.convolve(column, K, mode='same')
    return filtered_column


def fetchImage(RESULT_FOLDER= RESULT_FOLDER, returnBoth = False):
    K = np.loadtxt(f"{RESULT_FOLDER}/left/K_matrix.txt")
    R = np.loadtxt(f"{RESULT_FOLDER}/left/R_matrix.txt")
    T = np.loadtxt(f"{RESULT_FOLDER}/left/T_matrix.txt")

    plt.ion()

    left_images_filenames = list(filter(lambda fn: fn.split(".")[-1]=="png", os.listdir(f"{RESULT_FOLDER}/left")))
    timestamps = list(map(lambda fn: fn.split(".")[0], left_images_filenames))
    ti = 0
    timestamp = timestamps[ti]
    left = cv2.imread(f"{RESULT_FOLDER}/left/{timestamp}.png")
    disp = np.array(cv2.imread(f"{RESULT_FOLDER}/disp_zed/{timestamp}.png", cv2.IMREAD_ANYDEPTH) / 256.0, dtype=np.float32)
    if(returnBoth):
        return disp, left
    return disp

# Input should be a disparity image where each column is differentiated
def flatten_and_plot_histogram(df_derivative, plot=True,title="Histogram of Derivative Values", prompt=False, input_bins=2000):
    # Flatten the array of derivatives to a 1D array
    flattened_derivatives = df_derivative.flatten()

    if(plot):
        # Plot the histogram
        plt.figure()
        plt.hist(flattened_derivatives, bins=input_bins, range=(-0.3,0.5), color='blue', edgecolor='blue')
        plt.title(title)
        plt.xlabel('Derivative Value')
        plt.ylabel('Frequency')
        # Show the plot
        plt.show()
        plt.pause(5)
    
    if(prompt):
        guess1 = np.array([input("Please enter the initial guess: ")])
        guess2 = np.array([input("Please enter the second guess: ")])
        #initial_centers = np.array([guess1, guess2])

        return flattened_derivatives, guess1, guess2

    return flattened_derivatives


# Methods for clustering and graph fitting
def gaussian(x, mean, stddev):
    return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

# Gaussian fit function
def fit_gaussian(data):
    mean = np.mean(data)
    stddev = np.std(data)
    return mean, stddev

def clustering(flattened_derivatives, guess1=[0.0], guess2=[0.06], a_mini=-0.3, a_maxi=0.5, clusters=2):
    # Clipping the data to be between -0.3 and 0.5
    clipped_derivatives = np.clip(flattened_derivatives, a_min=a_mini, a_max=a_maxi)

    # Reshape data for KMeans
    clustering_model = clipped_derivatives.reshape(-1, 1)
    
    # Initial guesses for cluster centers (approximated from histogram peaks)
    initial_centers = np.array([guess1, guess2])  # replace peak1, peak2 with your estimates

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=clusters, init=initial_centers, n_init=1)
    kmeans.fit(clustering_model)

    # Get the labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    return clustering_model, clipped_derivatives, labels, centers, 

def curve_fitting(clustering_model, clipped_derivatives, labels, plot=True, k=2):
    # Separate the data into two clusters
    cluster1 = clustering_model[labels == 0]
    cluster2 = clustering_model[labels == 1]

    # Fit Gaussian to each cluster
    mean1, stddev1 = fit_gaussian(cluster1)
    mean2, stddev2 = fit_gaussian(cluster2)

    # Calculate range limits for each Gaussian curve
    dL1 = mean1 - k * stddev1
    dU1 = mean1 + k * stddev1
    dL2 = mean2 - k * stddev2
    dU2 = mean2 + k * stddev2

    if(plot):
        # Create a range of x values
        x_values = np.linspace(-0.3, 0.5, 400)

        # Gaussian curves
        gaussian_curve1 = gaussian(x_values, mean1, stddev1)
        gaussian_curve2 = gaussian(x_values, mean2, stddev2)

        # Plot histogram and Gaussian curves
        plt.figure()
        plt.hist(clipped_derivatives, bins=200, range=(-0.3, 0.5), density=True, alpha=0.6)
        plt.plot(x_values, gaussian_curve1, label='Gaussian 1')
        plt.plot(x_values, gaussian_curve2, label='Gaussian 2')
        plt.xlabel('Derivative Values')
        plt.ylabel('Frequency')
        plt.title('Histogram with Fitted Gaussian Curves')
        plt.legend()
        plt.show()
        plt.pause(5)
        
    return dL1, dU1, dL2, dU2

# Pixel classification
def pixel_classification(df_derivative, dL1, dU1, dL2, dU2):
    # And dL1, dU1, dL2, dU2 are the range limits for the two Gaussians
    classification_map = np.empty(df_derivative.shape, dtype=object)

    for i in range(df_derivative.shape[0]):  # Rows
        for j in range(df_derivative.shape[1]):  # Columns
            derivative_value = df_derivative[i, j]

            # Check if the derivative value falls into the range of either Gaussian
            if dL1 <= derivative_value <= dU1:
                classification = 'upright'
            elif dL2 <= derivative_value <= dU2:
                classification = 'horizontal'
            else:
                classification = 'unknown'

            classification_map[i, j] = classification

    return classification_map

# Define colors for each classification
colors = {
    'upright': [1, 0, 0],  # Red
    #'horizontal': [0, 0, 1],  # Blue
    'horizontal': [0, 1, 0],  # Green
    'unknown': [0.5, 0.5, 0.5]  # Gray
}

def assign_colours_and_plot(classification_map, colors=colors, plot=True, title="Pixel-wise Disparity Map Classification"):
    # Create an empty array for the color-coded image
    color_coded_image = np.zeros((*classification_map.shape, 3))

    # Assign colors
    for classification, color in colors.items():
        mask = classification_map == classification
        color_coded_image[mask] = color

    if(plot):
        plt.figure(figsize=(10, 6))
        plt.imshow(color_coded_image)
        plt.title(title)
        plt.axis('off')  # Hide the axes
        plt.show()
        plt.pause(5)

# Main method
def main():

    # Fetch the image
    disp = fetchImage()

    # Interpolate the columns of the disp image
    dI = interpolate_each_column(disp)


    height, width = dI.shape
    df = np.zeros_like(dI)  # Low-pass filtered disparity image
    for x in range(width):
        df[:, x] = low_pass_filter(dI[:, x])

    # Derivative and pad
    df_derivative = np.diff(df, axis=0)
    df_derivative = np.pad(df_derivative, ((0, 1), (0, 0)), mode='edge')

    flattened_derivatives, g1, g2= flatten_and_plot_histogram(df_derivative, title="Derivated histogram", prompt=True)
    #g1, g2 = [0.0], [0.07]
    print("Guess 1: ",g1, "Guess2: ", g2) 

    clustering_model, clipped_derivatives, labels, centers = clustering(flattened_derivatives, guess1=g1, guess2=g2)

    dL1, dU1, dL2, dU2 = curve_fitting(clustering_model=clustering_model, clipped_derivatives=clipped_derivatives, labels=labels)

    classification_map = pixel_classification(df_derivative, dL1, dU1, dL2, dU2)

    assign_colours_and_plot(classification_map, title="Horizontal and vertical pixel segmentation")

if __name__ == "__main__":
    main()

