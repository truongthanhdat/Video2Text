import argparse
import skvideo.io as skv
import skimage.io
import numpy as np
import featureExtraction.feature_extract as fe
from s2vt.s2vt_captioner import generateCaption
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='videoplayback.mp4')
parser.add_argument('--output', type=str, default='output.mp4')
parser.add_argument('--block', type=int, default=31)
args = parser.parse_args()

BLOCK = args.block

vgg16 = fe.CaffeFeatureExtractor(
            model_path="featureExtraction/vgg16_deploy.prototxt",
            pretrained_path="featureExtraction/vgg16.caffemodel",
            blob="fc7",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )

def load_image(image, color=True):
    img = skimage.img_as_float(image).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def fc7(image):
    image = load_image(image)
    return vgg16.extract_feature(image)

video = skv.vread(args.input)

with open('input.txt', 'w') as output:
    index = 1
    vid = 1
    for i in xrange(video.shape[0]):
        if (index == BLOCK):
            print 'Fisish shot', vid
            vid = vid + 1
            index = 1
        s = 'vid%d_frame_%d,' % (vid, index)
        feature = fc7(video[i, :, :, :]).tolist()
        s = s + ','.join([str('%0.9f' % x) for x in feature])
        s = s + '\n'
        output.write(s)
        index = index + 1

result = generateCaption(['input.txt'])

with open('output.txt', 'w') as output:
    for i in result:
        output.write('{0}:\t{1}\n'.format(i, result[i]))

vid = 1
index = 1
for i in xrange(video.shape[0]):
    if (index == BLOCK):
        vid = vid + 1
        index = 1

    cv2.putText(video[i], result['vid%d' % vid], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    index = index + 1

