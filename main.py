import argparse
import skvideo.io as skv
import numpy as np
from s2vt.s2vt_captioner import generateCaption
import VGG16.vgg16 as vgg16
import cv2
import tensorflow as tf
from scipy.misc import imresize

def convertImage(images):
    imgs = []
    for img in images:
        img = imresize(img, (224, 224))
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

#Analysis arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='videoplayback.mp4')
parser.add_argument('--output', type=str, default='output.mp4')
parser.add_argument('--block', type=int, default=30)
args = parser.parse_args()
BLOCK = args.block

#Reading video
video = skv.vread(args.input)

"""
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	#VGG-16 Initilization
	sess = tf.Session()
	input = tf.placeholder(tf.float32, [None, 224, 224, 3])
	vgg = vgg16.vgg16(input, 'VGG16/vgg16_weights.npz', sess)

	#Feature Extracction
	with open('input.txt', 'w') as output:
	    first = 0
	    vid = 1
	    while (first < video.shape[0]):
		last = np.min([first + BLOCK, video.shape[0]])
		images = convertImage(video[first:last])
		features = vgg.fc7(sess, images)

		index = first
		s = '\n'.join([str('vid%d_frame_%d,' % (vid, frame)) + ','.join([str('%0.9f' % x) for x in features[frame]]) for frame in xrange(last - first)])
		output.write(s + "\n")

		print 'Finish shot', vid
		vid = vid + 1
		first = last
"""
#Generating Caption
result = generateCaption(['input.txt'])
with open('output.txt', 'w') as output:
    for i in result:
        output.write('{0}:\t{1}\n'.format(i, result[i]))

#Writing Video
vid = 1
index = 1
for i in xrange(video.shape[0]):
    if (index == BLOCK + 1):
        vid = vid + 1
        index = 1

    cv2.putText(video[i], result['vid%d' % vid], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    index = index + 1
skv.vwrite(args.output, video)
