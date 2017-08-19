import argparse
import skvideo.io as skv
import skimage.io
import numpy as np
import caffe
import featureExtraction.feature_extract as fe
from s2vt.s2vt_captioner import generateCaption
import os
import time
import progressbar

##---------Analysis-Arguments------------------##
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Video input file', default='vbs.mp4')
parser.add_argument('--output', type=str, help='Video output file', default='output.mp4')
parser.add_argument('--shot', type=int, help='Frames per shot', default=30)
parser.add_argument('--step', type=int, help='Step size', default=30)
args = parser.parse_args()
##---------------------------------------------##


##-------Constant-Initialization---------------##
SHOT = args.shot
STEP = args.step
##---------------------------------------------##


##--------Loading-Data-------------------------##
#Loading Model VGG16
vgg16 = fe.CaffeFeatureExtractor(
            model_path="featureExtraction/vgg16_deploy.prototxt",
            pretrained_path="featureExtraction/vgg16.caffemodel",
            blob="fc7",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )

#Loading Video
video = skv.vread(args.input)
##---------------------------------------------##


##---------Declare-Function--------------------##
#Converting Image
def load_image(image, color=True):
    img = skimage.img_as_float(image).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

#Extracting Image
def fc7(image):
    skimage.io.imsave('tmp.jpg', image)
    image = caffe.io.load_image('tmp.jpg')
    os.remove('tmp.jpg')
    return vgg16.extract_feature(image)

def featureExtraction(video):
    features = []
    bar = progressbar.ProgressBar()
    index = bar(xrange(video.shape[0]))
    for i  in index:
        features.append(fc7(video[i]))
    features = np.array(features)
    return features

##---------------------------------------------##


##---------Feature-Extraction------------------##
startTime = time.time()
features = featureExtraction(video)
endTime = time.time()
with open('time.txt', 'w') as output:
    output.write('Feature extraction takes %0.10f seconds\n'% (endTime - startTime))


#Writing Features
with open('input.txt', 'w') as output:
    first = 0
    index = 0
    while (first < video.shape[0]):
        last = np.min([video.shape[0], first + SHOT])
        subFeatures = features[first:last]
        s = '\n'.join([str('vid%d_frame_%d,' % (index, frame)) + ','.join([str('%0.9f' % x) for x in subFeatures[frame]]) for frame in xrange(last - first)])
        output.write(s + '\n')
        print 'Finish shot', index
        index = index + 1
        first = first + STEP

##---------------------------------------------##


##--------Generating-Caption-------------------##
#Generating Caption
startTime = time.time()
result = generateCaption(['input.txt'])
endTime = time.time()
with open('time.txt', 'a') as output:
    output.write('Generating caption takes %0.10f seconds\n' % (endTime - startTime))

with open('output.txt', 'w') as output:
    for i in result:
        output.write('{0}:\t{1}\n'.format(i, result[i]))
##---------------------------------------------##
