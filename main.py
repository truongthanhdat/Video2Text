import argparse
import skvideo.io as skv
import skimage.io
import numpy as np
import featureExtraction.feature_extract as fe
from s2vt.s2vt_captioner import generateCaption
import cv2

#Analysis Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Video input file', required=True)
parser.add_argument('--output', type=str, help='Video output file', default='output.mp4')
parser.add_argument('--shot', type=int, help='Frames per shot',default=30)
args = parser.parse_args()
SHOT = args.shot

#Load Model VGG16
vgg16 = fe.CaffeFeatureExtractor(
            model_path="featureExtraction/vgg16_deploy.prototxt",
            pretrained_path="featureExtraction/vgg16.caffemodel",
            blob="fc7",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )

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
    image = load_image(image)
    return vgg16.extract_feature(image)

#Loading Video (B, G, R)
video = skv.vread(args.input)

#Writing Features
with open('input.txt', 'w') as output:
    index = 0
    vid = 1
    for i in xrange(video.shape[0]):
        if (index == SHOT):
            print 'Finish shot', vid
            vid = vid + 1
            index = 0

        s = 'vid%d_frame_%d,' % (vid, index)
        feature = fc7(video[i, :, :, :]).tolist()
        s = s + ','.join([str('%0.9f' % x) for x in feature])
        s = s + '\n'
        output.write(s)
        index = index + 1

#Generating Caption
result = generateCaption(['input.txt'])
with open('output.txt', 'w') as output:
    for i in result:
        output.write('{0}:\t{1}\n'.format(i, result[i]))

#Save Video
vid = 1
index = 0
for i in xrange(video.shape[0]):
    if (index == SHOT):
        vid = vid + 1
        index = 0

    cv2.putText(video[i], result['vid%d' % vid], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    index = index + 1

skv.vwrite(args.output, video)
