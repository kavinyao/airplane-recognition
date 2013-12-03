import numpy as np
from SimpleCV.ImageClass import Image
from SimpleCV.Features.FeatureExtractorBase import FeatureExtractorBase

class BasicFeatureByGridExtractor(FeatureExtractorBase):
    """Detect lines, corners and keypoints in given image and count by
    3x3 grids. Return normalized result."""

    FEATURES = ['Line', 'Corner', 'Keypoint']

    def __init__(self, grid_size=3):
        self.grid_size = 3

    def extract(self, img):
        features = []
        lines = img.findLines()
        features.extend(self._generate_feature(lines, img.width, img.height))
        corners = img.findCorners()
        features.extend(self._generate_feature(corners, img.width, img.height))
        key_points = img.findKeypoints()
        features.extend(self._generate_feature(key_points, img.width, img.height))

        return features

    def _generate_feature(self, detected_objects, width, height):
        width_unit = width / self.grid_size + 1
        height_unit = height / self.grid_size + 1

        features = np.zeros(self.grid_size*self.grid_size, dtype='float')
        for ob in detected_objects:
            idx = (int(ob.x) / width_unit) + (int(ob.y) / height_unit) * self.grid_size
            features[idx] += 1

        # normalize
        return features / features.max()

    def getFieldNames(self):
        return ['Basic-Grid-%s-%d' % (fn, i) for fn in self.FEATURES\
            for i in range(self.grid_size*self.grid_size)]

    def getNumFields(self):
        return len(self.FEATURES) * self.grid_size * self.grid_size

if __name__ == '__main__':
    import os
    import sys
    from glob import glob

    fe = BasicFeatureByGridExtractor()

    img_dirs = sys.argv[1:]
    avg_feature_vectors = []
    for img_dir in img_dirs:
        img_files = glob(os.path.join(img_dir, '*.jpg'))

        features = np.zeros(fe.grid_size*fe.grid_size*len(fe.FEATURES))
        for img_file in img_files:
            img = Image(img_file)
            features += fe.extract(img)

        avg_feature_vectors.append(features / len(img_files))

    for i in range(len(avg_feature_vectors)):
        for j in range(len(avg_feature_vectors)):
            print '%.4f' % np.linalg.norm(avg_feature_vectors[i]-avg_feature_vectors[j]),
        print
