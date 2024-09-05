
import cv2
import numpy as np
import matplotlib.pyplot as plt
# scale bar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# find beads in image
def findbeadsinimage(false_color, beadsize=15, skip = [-1], maxnum = 10, label_bead = False, show_gray = False, colindices = [], use_colors = [], gray_thresh = 20):
    # Create a numpy array
    image_array = (false_color*255).astype('uint8')
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Convert the numpy array to grayscale
    gray = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary =  cv2.threshold(gray, gray_thresh, 255, cv2.THRESH_BINARY)
    if show_gray:
        plt.figure()
        plt.imshow(binary,cmap = 'gray')
        plt.colorbar()

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to remove small noise
    min_area = beadsize  # Minimum area of a bead
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    bead_locations = []
    for cnt in filtered_contours:
        # Calculate the centroid of the contour
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Append the centroid coordinates to the bead_locations list
        bead_locations.append((cx, cy))

    # Draw boxes around the centroids and label bead number
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    rgb_colors = [tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for color in colors]
    colindex  = 0
    if len(colindices) == 0: # if the color order is not specified, use default order
        colindices = range(len(rgb_colors))
    for i, (cx, cy) in enumerate(bead_locations):
        # Draw a rectangle around the centroid
        rad = 16
        x, y, w, h = cx-rad, cy-rad, 2*rad, 2*rad
        if i<maxnum and i not in skip:
            if use_colors==[]: # if specific colors aren't specified, use default colors
                cv2.rectangle(image_array_bgr, (x, y), (x+w, y+h), tuple(rgb_colors[colindices[colindex%len(colindices)]][::-1]), 2)
            else:
                cv2.rectangle(image_array_bgr, (x, y), (x+w, y+h), use_colors[colindices[colindex]], 2)
            colindex += 1
        #Label the bead number
        if label_bead:
            cv2.putText(image_array_bgr, f"Bead {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return (bead_locations[:maxnum], np.array(cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB))/255)


def drawscalebar(ax, scalebarval, pxlsize, mag = 1, scalebarname = '1 superpixel', loc = 'upper left', size_vertical = 10):
    scalebarsize = scalebarval*mag # micron
    scalepix = int(scalebarsize/pxlsize)

    fontprops = fm.FontProperties(size=24)
    scalebar = AnchoredSizeBar(ax.transData,
                            scalepix, scalebarname, loc, 
                            pad=0.5, sep = 10,
                            color='white',
                            frameon=False,
                            size_vertical=size_vertical,
                            fontproperties=fontprops)
    return scalebar

def select_and_average_bands(data_cube, wavelengths, spectral_ranges):
    averaged_bands = []
    for spectral_range in spectral_ranges:
        # Find the band indices within the spectral range
        indices = np.where((wavelengths >= spectral_range[0]) & (wavelengths <= spectral_range[1]))[0]
        # Average the selected bands
        if len(indices) > 0:
            averaged_band = np.mean(data_cube[indices,:, :], axis=0)
        else:
            # If no bands fall within the range, use a dummy band of zeros
            averaged_band = np.zeros_like(data_cube[0,:, :])
        averaged_bands.append(averaged_band)
    # Stack the averaged bands along the last dimension to form an RGB image
    rgb_image = np.stack(averaged_bands, axis=0)
    rgb_image = np.transpose(rgb_image, (1,2,0))
 # make color the last dimension
    return rgb_image

def color_visualize(image, wavelengths, title='', figsize=(10,10)):
    # Spectral ranges for averaging (in nm) for RGB channels
    spectral_ranges = [ (570, 700),(495, 570), (450, 495)]

    # Select and average the bands
    rgb_image = select_and_average_bands(image, wavelengths, spectral_ranges)

    # Normalize the RGB image to enhance contrast if necessary
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

    # Display the image
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb_image)
    plt.title('False Color RGB Visualization with Averaged Bands')
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    return rgb_image