import argparse
import skvideo.io as skv
import numpy as np
from s2vt.s2vt_captioner import generateCaption
import VGG16.vgg16 as vgg16
import tensorflow as tf
from scipy.misc import imresize
import time

##---------Declare-Function------------------------##
def convertImage(images):
    imgs = []
    for img in images:
        img = imresize(img, (224, 224))
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

def featureExtraction(images):
    first = 0
    BATCH_SIZE = 10
    features = []
    N = images.shape(0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        input = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16.vgg16(input, 'VGG16/vgg16_weights.npz', sess)

        while (first < N):
            last = np.min([N, first + BATCH_SIZE])
            feats = vgg.fc7(convertImage(images[first:last]))
            features.append(feats)
            first = last

    return np.array(features)


##-------------------------------------------------##


##----------Analysis-Arguments---------------------##
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Input video', default='vbs.mp4')
parser.add_argument('--output', type=str, help='Output video', default='output.mp4')
parser.add_argument('--shot', type=int, help='Frames per shot', default=30)
parser.add_argument('--step', type=int, help='Step size per shot', default=30)
args = parser.parse_args()
##-------------------------------------------------##


##---------Constant-Initilization------------------##
SHOT = args.shot
STEP = args.step
##-------------------------------------------------##


##--------Feature-Extration------------------------##
#Reading video
video = skv.vread(args.input)

startTime = time.time()
features = featureExtraction(video)
endTime = time.time()
with open('time.txt', 'w') as output:
    output.write('Feature extraction takes %0.10f seconds\n' % (endTime - startTime))

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
##-------------------------------------------------##


##-------------------------------------------------##
#Generating Caption
startTime = time.time()
result = generateCaption(['input.txt'])
endTime = time.time()
with open('time.txt', 'a') as output:
    output.write('Generating caption takes %0.10f seconds\n' % (endTime - startTime))

with open('output.txt', 'w') as output:
    for i in result:
        output.write('{0}:\t{1}\n'.format(i, result[i]))
##-------------------------------------------------##

