# Imports
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from segmentation import blur_object_pixels
from normalize_HSV import normalize_brightness, apply_mask, np2img
from draw_outlines import draw_outlines
import cv2

def full_pipeline(image_path, segmentation_model='deeplabv3_resnet101', blur_sigma=3, processing_scale=1, brightness_segments=4, canny_lower_threshold=0, canny_upper_threshold=200, line_width=1):
    filename = image_path  # path to your image
    input_image = Image.open(filename)  # open the image with PIL
    input_image = input_image.convert("RGB")  # convert to RGB format
    model = torch.hub.load('pytorch/vision:v0.10.0', segmentation_model, pretrained=True)  # load the deeplabv3_resnet101 model
    input_image_blurred, person_mask = blur_object_pixels(model, input_image, ['person'], sigma=blur_sigma, show_object_list=False, concat=False, scale=processing_scale)  # get the segmented image and the person mask
    ms = normalize_brightness(image=input_image_blurred, N=brightness_segments, object_mask=person_mask)  # normalize brightness into N different levels
    masked = apply_mask(input_image, mask = ms["all"])  # get the complete brightness mask
    img = np.array(masked)[:, :, ::-1].copy()  # placeholder for final image
    edges = Image.fromarray(cv2.Canny(image=img, threshold1=canny_lower_threshold, threshold2=canny_upper_threshold))  # detect edges in img using canny from openCV
    outlines = Image.composite(edges, person_mask, person_mask)  # only get edges in person segment
    final_image = np2img(draw_outlines(masked,outlines, line_width=line_width))  # color in lines in black
    display(input_image)  # show input image
    display(final_image)  # show final image