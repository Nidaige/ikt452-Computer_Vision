import torch
from torchvision import transforms
from PIL import Image, ImageChops
import os

def blur_object_pixels(model, input_image, objects_to_blur, sigma=1.0, show_object_list=False, concat=False, scale=1):
    """
    This function blurs the pixels of the objects listed in 'objects_to_blur' on the input 'input_image'
    using the semantic segmentation 'model' and returns the modified image. The amount of blur is
    controlled by the parameter 'sigma'.
    """
    
    model.eval()
    
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    
    objects = {
        0:'background',
        1:'aeroplane',
        2:'bicycle',
        3:'bird',
        4:'boat',
        5:'bottle',
        6:'bus',
        7:'car',
        8:'cat',
        9:'chair',
        10:'cow',
        11:'dining table',
        12:'dog',
        13:'horse',
        14:'motorbike',
        15:'person',
        16:'potted plant',
        17:'sheep',
        18:'sofa',
        19:'train',
        20:'tv/monitor'
    }
    
    objects_segmented = {}
    
    for output_prediction in output_predictions.unique().tolist():
        objects_segmented[output_prediction] = objects[output_prediction]
    
    if show_object_list:
        print(f'objects: {list(objects_segmented.values())}')
        
    
    mask = torch.zeros_like(output_predictions, dtype=torch.uint8)
    
    for key in objects_segmented.keys():
        if objects_segmented[key] in objects_to_blur:
            index = (output_predictions == key)
            mask[index] = 255
    
    input_image_blurred = transforms.GaussianBlur(kernel_size=(3,3), sigma=sigma)(input_image)
    
    mask = transforms.ToPILImage()(mask).convert('L')
    '''
    At this point, mask refers to the detected object
    - We apply our transforms/effects to the WHOLE image, then composite the altered image with the original, using the mask as seen below.
    - We can probably apply any number of compositions here, using the result of one transform as the input to the next, building our finished product
    - This is probably where we will have to call functions to apply to image.
    '''
    masked_image = Image.composite(input_image_blurred, input_image, mask)
    
    output_image = masked_image.copy()
    if concat:
        output_image = Image.new('RGB', (input_image.width + masked_image.width, max(input_image.height, masked_image.height)))
        output_image.paste(input_image, (0, 0))
        output_image.paste(masked_image, (input_image.width, 0))
    
    if int(scale) > 1:
        (width, height) = (output_image.width // int(scale), output_image.height // int(scale))
        output_image = output_image.resize((width, height))
            
    return output_image