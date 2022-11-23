import numpy as np
import math
from PIL import Image, ImageChops


def normalize_brightness(image, N, object_mask):
    '''
    Takes in a PIL image object, and returns a dictionary of N masks that correspond to areas within certain brightness/Value (HSV)-brackets
    '''
    o_mask = np.array(object_mask)
    hsv = RGB_to_HSV(image)
    brightness_brackets = list(np.linspace(0,1,N))
    masks = {}
    for bracket in brightness_brackets:
            masks[str(brightness_brackets.index(bracket))] = np.zeros(hsv.shape[0:-1])
            masks["all"] = np.zeros(hsv.shape[0:-1])
    
    for y in range(hsv.shape[0]):
        for x in range(hsv.shape[1]):
            H, S, V = hsv[y, x, :]
            applied = False
            for bracket in brightness_brackets:
                if V <= bracket and not applied:
                    applied = True
                    masks[str(brightness_brackets.index(bracket))][y,x] = min(255, o_mask[y,x])
                    masks["all"][y,x] = min(255, o_mask[y,x])*bracket
    for mask in masks.keys():
        masks[mask] = Image.fromarray((masks[mask]).astype(np.uint8))
    return masks
                    
    
    
def RGB_to_HSV(RGB):
    '''
    Performs RGB to HSV conversion on the input Image object RGB and
    returns HSV representation as a numpy array.
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
    # Student name: Henning Blomfeldt Thorsen
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
    