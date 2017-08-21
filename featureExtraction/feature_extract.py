import numpy as np
import caffe

class CaffeFeatureExtractor:
    def __init__(self, model_path, pretrained_path, blob, crop_size, meanfile_path=None, mean_values=None):
        caffe.set_mode_gpu()
        self.model_path = model_path
        self.pretrained_path = pretrained_path
        self.blob = blob
        self.crop_size = crop_size
        self.meanfile_path = meanfile_path
        self.mean_values = mean_values
        # create network
        self.net = caffe.Net(self.model_path, self.pretrained_path, caffe.TEST)
        # mean
        if self.meanfile_path is not None:
            # load mean array
            self.mean = np.load(self.meanfile_path) # expect that shape = (1, C, H, W)
            self.mean = self.mean[0]
            self.mean = self.crop_matrix(self.mean, crop_size=self.crop_size)
        elif self.mean_values is not None:
            # create mean array
            assert len(self.mean_values) == 3
            self.mean = np.zeros((3, self.crop_size, self.crop_size))
            self.mean[0] = mean_values[0]
            self.mean[1] = mean_values[1]
            self.mean[2] = mean_values[2]
        else:
            raise Exception

        self.transformer = caffe.io.Transformer({"data": self.net.blobs["data"].data.shape}) # for cropping
        self.transformer.set_transpose("data", (2,0,1)) # (H,W,C) => (C,H,W)
        self.transformer.set_mean("data", self.mean) # subtract by mean

    def extract_feature(self, imgs):
        preprocessed_imgs = np.zeros((len(imgs), 3, 224, 224), dtype=np.float32)
        for i in xrange(len(imgs)):
            preprocessed_imgs[i] = self.transformer.preprocess("data", imgs[i])

        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_imgs, "blobs": [self.blob]})
        feat = out[self.blob]
        return feat

    def crop_matrix(self, matrix, crop_size):
        """
        :param matrix numpy.ndarray: matrix, shape = [C,H,W]
        :param crop_size integer: cropping size
        :return: cropped matrix
        :rtype: numpy.ndarray, shape = [C,H,W]
        """
        assert matrix.shape[1] == matrix.shape[2]
        corner_size = matrix.shape[1] - crop_size
        corner_size = np.floor(corner_size / 2)
        res = matrix[:, corner_size:crop_size+corner_size, corner_size:crop_size+corner_size]
        return res
