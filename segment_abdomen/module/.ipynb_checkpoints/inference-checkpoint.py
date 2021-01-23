import pydicom
import numpy as np
import cv2
import torch
import segmentation_models_pytorch
import sys


def load_image(img_path):
    image = pydicom.dcmread(f"{img_path}").pixel_array
    image = np.array(image)
    return image
    
def preprocessing(image):
    # set margin
    margin = 70    
    image = image[margin:-margin,margin:-margin]
    # upper, lower threshold
    under_val = 700
    over_val = 1200
    over_idx  = (image > over_val)
    under_idx = (image < under_val)
    image[over_idx]  = over_val
    image[under_idx] = under_val
    image = (image - under_val) / (over_val - under_val) * 255
    image = image.astype(np.uint8)[..., np.newaxis]
    # resize image to (512,512,1)
    image = cv2.resize(image, (512, 512))[...,np.newaxis]
    # to tensor
    image = image.transpose(2, 0, 1).astype('float32')[np.newaxis,...]
    image_tensor = torch.cuda.FloatTensor(image)
    return image_tensor

def predict(image_tensor, model):
    predict_mask = model.predict(image_tensor).cpu().numpy()[0,:,:,:]
    predict_mask = predict_mask.transpose([1,2,0]).argmax(axis = 2)
    return predict_mask


if __name__ == "__main__":
    # input path
    if sys.argv[1] == '-img_path':
        img_path = sys.argv[2]
    if sys.argv[3] == '-model_path':
        model_path = sys.argv[4]

    #set path, model
    model = torch.load(model_path)
    # load image
    image = load_image(img_path)
    # preprocessing
    image_tensor = preprocessing(image)
    # predict
    predict_mask = predict(image_tensor, model)
    
    print((predict_mask == 0).sum() / len(predict_mask.reshape(-1)) * 100)
    print((predict_mask == 1).sum() / len(predict_mask.reshape(-1)) * 100)
    print((predict_mask == 2).sum() / len(predict_mask.reshape(-1)) * 100)
    print((predict_mask == 3).sum() / len(predict_mask.reshape(-1)) * 100)
    return predict_mask