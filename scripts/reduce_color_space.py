import numpy as np
from SimpleCV.ImageClass import Image
from sklearn.cluster import KMeans

def reduce_color_space(img, N=8, fast=False):
    """Reduce color space to N colors using k-means.
    Note: assume RGB space.
    @param img instance of Image
    @param fast trade in accuracy for performance
    @return new Image
    """
    img_data = img.getNumpy()
    width, height, _ = img_data.shape
    if width == img.size()[1]:
        # strange, it's rotated (column-based)
        img_data = img_data.swapaxes(0, 1)
    colors = img_data.reshape(width*height, 3).astype('float')
    # get centroid colors
    kmeans = KMeans(n_clusters=N, n_init=1 if fast else 3)
    kmeans.fit(colors)
    centroid_colors = kmeans.cluster_centers_.astype('uint8')
    assignments = kmeans.predict(colors)
    # replace similar colors with centroid colors
    for i in range(N):
        # multi indices simultaneous assignment
        colors[(assignments == i).nonzero()[0]] = centroid_colors[i];

    return Image(colors.reshape(img_data.shape))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print 'Usage: python %s <image_file> [output_file="output.jpg"]' % sys.argv[0]
        sys.exit(1)

    output = 'output.jpg' if len(sys.argv) < 3 else sys.argv[2]

    new_img = reduce_color_space(Image(sys.argv[1]))
    new_img.save(output)
