import feature_extract as fe
import skimage.io
import numpy as np

def load_image(image, color=True):
    img = skimage.img_as_float(image.astype(np.float32))
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def fc7(image):
    vgg16 = fe.CaffeFeatureExtractor(
            model_path="vgg16_deploy.prototxt",
            pretrained_path="vgg16.caffemodel",
            blob="fc7",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )
    image = load_image(image)
    return vgg16.extract_feature(image)


