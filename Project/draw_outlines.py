import math
import numpy as np

def draw_outlines(im, edges, line_width):
    '''
    Takes in PIL images im and edges and integer line_width, and uses <edges> to draw lines into <im> with a width of <line_width>
    '''

    img_with_lines = np.array(im.copy())  # placeholder for final image. Copy of segmented, normalized image
    outlines_arr = np.array(edges)  # make a numpy array of detected edges for easier iteration
    for y_coord in range(outlines_arr.shape[0]):  # for each y-pixel
        for x_coord in range(outlines_arr.shape[1]):  # for each x-pixel
            if outlines_arr[y_coord, x_coord] > 5:  # if the detected edge has a value greater than 5 (out of 255):
                # Set the pixel at (y,x) +- half the line width to 0
                img_with_lines[max(y_coord-math.floor(0.5*line_width),0):min(y_coord+math.ceil(0.5*line_width), outlines_arr.shape[0]),max(x_coord-math.floor(0.5*line_width),0):min(x_coord+math.ceil(0.5*line_width), outlines_arr.shape[1])] = 0
    return img_with_lines
