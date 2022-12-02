import numpy as np
from PIL import Image


def normalize_brightness(image, N, object_mask):
    '''
    Takes in a PIL image object, an integer N, and another PIL image object object_mask, and returns a dictionary of N masks that correspond to areas within certain brightness/Value (HSV)-brackets.
    '''
    o_mask = np.array(object_mask)  # turn mask into numpy array
    hsv = RGB_to_HSV(image)  # convert image to hsv
    brightness_brackets = list(np.linspace(0,1,N))  # divide the values from 0 to 1 into N equally sized intervals
    masks = {}  # dictionary to hold the masks for each brightness interval
    for bracket in brightness_brackets:  # for each interval:
            masks[str(brightness_brackets.index(bracket))] = np.zeros(hsv.shape[0:-1])  # initialize the masks for each interval
            masks["all"] = np.zeros(hsv.shape[0:-1])  # also initialize the mask for all the intervals put together
    
    for y in range(hsv.shape[0]):  # for each y-pixel:
        for x in range(hsv.shape[1]):  # for each x-pixel:
            H, S, V = hsv[y, x, :]  # get the HSV-value for pixel at(y,x)
            applied = False  # pixel has not been applied yet
            for bracket in brightness_brackets:  # for each interval:
                if V <= bracket and not applied:  # if the pixel Value is greater than the current interval, and has not been applied yet:
                    applied = True  # has now been applied
                    masks[str(brightness_brackets.index(bracket))][y,x] = min(255, o_mask[y,x])  # update the pixel (y,x) in the mask corresponding to the current interval
                    masks["all"][y,x] = min(255, o_mask[y,x])*bracket  # update the pixel (y,x) in the mask representing all the intervals
    for mask in masks.keys():  # for each mask in the dictionary,
        masks[mask] = Image.fromarray((masks[mask]).astype(np.uint8))  # convert to a uint8 numpy array
    return masks
                    
    
    
def RGB_to_HSV(RGB):
    '''
    Performs RGB to HSV conversion on the input Image object RGB and
    returns HSV representation as a numpy array.
    
    This functionw as supplied as part of Practical One.
    '''
    RGB = np.array(RGB).astype(float) / 255.0
    HSV = np.zeros(RGB.shape)
    for y in range(RGB.shape[0]):
        for x in range(RGB.shape[1]):
            R, G, B = RGB[y,x,:]
            V = np.max((R, G, B))
            m = np.min((R, G, B))
            C = V - m
            if V == 0.0:
                S = 0.0
            else:
                S = C / V
                
            if C == 0.0:
                H = 0.0
            else:
                if V == R:
                    H_prime = (G-B) / C
                elif V == G:
                    H_prime = (B-R) / C + 2
                elif V == B:
                    H_prime = (R-G) / V + 4
                
                if H_prime < 0:
                    H = H_prime / 6 + 1
                else:
                    H = H_prime / 6
            
            HSV[y,x,0] = H
            HSV[y,x,1] = S
            HSV[y,x,2] = V
    
    return HSV


def HSV_to_RGB(HSV):
    '''
    Performs HSV to RGB conversion on the input numpy array HSV and
    returns RGB representation as a PIL image object

    Developing this function was part of Practical One.
    '''
    import math
    rgb = np.zeros(HSV.shape)  # array to hold the rgb-values
    for y_coord in range(HSV.shape[0]):  # loop through x coordinates
        for x_coord in range(HSV.shape[1]):  # Loop through Y-coordinates
            H, S, V = HSV[y_coord, x_coord, :]  # extract HSV - values
            # Following the algorithm put forth in the paper "Color Gamut Transform Pairs" by Alvy Ray Smith:
            H *= 6 # H := 6H
            I = math.floor(H)  # Let I := floor(H)
            F = H-I  # F := H-I
            M = V*(1-S)  # M := V*(1-S)
            N = V*(1-(S*F))  # N := V*(1-S*F)
            K = V*(1-S*(1-F))  # K := V*(1-S*(1-F))
            r,g,b = [(V,K,M),(N,V,M),(M,V,K),(M,N,V),(K,M,V),(V,M,N)][I]  # switch case of RGB based on value of I
            # values are decimals between 0 and 1, multiply by 255 to get valid RGB-values, and insert into image
            rgb[y_coord, x_coord, :] = [r*255, g*255, b*255]
    return Image.fromarray((rgb).astype(np.uint8))  # return image with type uint8 for display purposes


def apply_mask(im, mask):
    '''
    Takes two image objects, im and mask, converts im to HSV and makes the V-values match the non-zero values in mask, before returning a PIL image object rgb.
    '''
    hsv_im = RGB_to_HSV(im)
    hsv_mask = np.array(mask)
    for y in range(hsv_im.shape[0]):
        for x in range(hsv_im.shape[1]):
            H, S, V = hsv_im[y,x,:]
            V_mask = hsv_mask[y,x] / 255
            if V_mask > 0:
                hsv_im[y,x] = (H,S,V_mask)
            else:
                hsv_im[y,x] = (H,S,V)
    rgb = HSV_to_RGB(hsv_im)
    return rgb
    
def np2img(im, norm=False, rgb_mode=False):
    """
    This function converts the input numpy object im to Image object and returns
    the converted object. If norm == True, then the input is normalised to [0,1]
    using im <- (im - im.min()) / (im.max() - im.min()).

    This function was supplied with Practical Three
    """
    if norm:
        if ((im.max() - im.min()) != 0.0):
            im = (im - im.min()) / (im.max() - im.min())

    if ((im.min() >= 0.0) and (im.max() <= 1.0)):
        im = im * 255.0

    if rgb_mode and im.ndim == 2:
        im = im[...,np.newaxis].repeat(3, axis=2)
        
    if im.ndim == 2:
        im = Image.fromarray(im.astype(np.uint8), mode='L')
    elif (im.ndim == 3) and (im.shape[2] == 3):
        im = Image.fromarray(im.astype(np.uint8), mode='RGB')

    return im
