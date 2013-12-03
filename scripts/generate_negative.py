import os
import sys
import random
from os import path
from glob import glob
from SimpleCV import Image

"""Generate 600 negative images examles for Haar feature training."""

N = 200
MAX_WIDTH = 800

def generate_negative_examples(argv):
    image_dirs = argv[4:]

    images = []
    for image_dir in image_dirs:
        # grab all images
        images.extend(glob(path.join(image_dir, '*.jpg')))
        images.extend(glob(path.join(image_dir, '*.JPG')))
        images.extend(glob(path.join(image_dir, '*.png')))
        images.extend(glob(path.join(image_dir, '*.PNG')))

    images = set(images)

    if len(images) < N:
        print 'Not enough images! (got %d, need %d)' % (len(images), N)
        return

    width, height, output_dir = int(argv[1]), int(argv[2]), argv[3]

    if path.exists(output_dir) and (not path.isdir(output_dir)):
        print '%s is not a directory' % output_dir
        return
    elif not path.exists(output_dir):
        os.mkdir(output_dir)

    for i in xrange(N):
        print 'generating %3d/%d...' % ((i+1), N)
        img = Image(images.pop())
        img = img.grayscale()
        if img.width > MAX_WIDTH:
            img = img.resize(MAX_WIDTH, int(1.0*img.height*MAX_WIDTH/img.width))

        x, y = random.randint(0, img.width-width), random.randint(0, img.height-height)
        img = img.crop(x, y, width, height)

        path_to_save = path.join(output_dir, '%d.png' % (i+1))
        img.save(path_to_save)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: python %s <width> <height> <output_dir> <image_dir> [<image_dir>, ...]' % sys.argv[0]
        sys.exit(1)

    generate_negative_examples(sys.argv)
