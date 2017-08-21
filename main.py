import argparse
import skvideo.io as skv
import numpy as np
import featureExtraction.feature_extract as fe
from s2vt.s2vt_captioner import generateCaption
import time
import progressbar

##---------Analysis-Arguments------------------##
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Video input file', default='vbs.mp4')
parser.add_argument('--output', type=str, help='Video output file', default='output.mp4')
parser.add_argument('--shot', type=int, help='Frames per shot', default=30)
parser.add_argument('--step', type=int, help='Step size', default=30)
parser.add_argument('--batchSize', type=int, help='Batch size', default=16)
args = parser.parse_args()
##---------------------------------------------##


##-------Constant-Initialization---------------##
SHOT = args.shot
STEP = args.step
BATCH_SIZE = args.batchSize
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

#Extracting Image
def fc7(images):
    return vgg16.extract_feature(images)

def featureExtraction(video):
    features = []
    bar = progressbar.ProgressBar()
    N = video.shape[0]
    index = bar(xrange((N - 1) / BATCH_SIZE + 1))
    first = 0
    for i  in index:
        last = np.min([N, first + BATCH_SIZE])
        feats = fc7(video[first:last])
        for feat in feats:
            features.append(feat)
        first = last

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

vid = -1
import cv2
for i in xrange(video.shape[0]):
    if ((i % SHOT) == 0):
        vid = vid + 1
    vidstr = 'vid%d' % vid
    cv2.putText(video[i], result[vidstr], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

skv.vwrite(args.output, video)
