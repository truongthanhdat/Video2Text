# Video to Text

# Usage:

## Download pretrained VGG-16

+ cd featureExtraction

+ wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

+ mv VGG_ILSVRC_16_layers.caffemodel vgg16.caffemodel

## Download pretraind S2VT

+ cd s2vt

+ mkdir snapshots

+ wget --no-check-certificate https://www.dropbox.com/s/wn6k2oqurxzt6e2/s2s_vgg_pstream_allvocab_fac2_iter_16000.caffemodel

+ mv s2s_vgg_pstream_allvocab_fac2_iter_16000.caffemodel snapshots/s2vt_vgg_rgb.caffemodel

## Generating captions:

+ python main.py --input input-video --output --output-video --block frames-per-shot

# References:

## [S2VT](https://gist.github.com/vsubhashini/38d087e140854fee4b14)

## [Feature Extraction](https://github.com/colingogo/caffe-pretrained-feature-extraction)

# Author:

## Thanh-Dat Truong. Email: thanhdattrg@gmail.com
