import os
import random
from os import path
from SimpleCV.ImageClass import Image
from collections import defaultdict
from optparse import OptionParser, OptionValueError
from reduce_color_space import reduce_color_space

FGVC_IMAGE_SUFFIX = 'jpg'
FGVC_BANNER_HEIGHT = 20

NORMALIZED_WIDTH = 300

def write_images(model_path, fgvc_data_dir, images_nos, options):
    if options.crop:
        # read box information
        box_info = {}
        with open(path.join(fgvc_data_dir, 'images_box.txt')) as ibf:
            for line in ibf:
                result = line.split()
                img_no, coords = result[0], map(int, result[1:])
                box_info[img_no] = coords

    for image_no in images_nos:
        image_file = '%s.%s' % (image_no, FGVC_IMAGE_SUFFIX)
        img = Image(path.join(fgvc_data_dir, 'images', image_file))

        if options.crop:
            # although FGVC pages says the orgin is (1,1)
            # still there's some coordinates in box have 0 value
            # so don't do adjustment - 1px difference should be OK
            x1, y1, x2, y2 = box_info[image_no]
            img = img.crop(x1, y1, x2, y2)
        else:
            # remove out banner
            img = img.crop(0, 0, img.width, img.height-FGVC_BANNER_HEIGHT)

        if options.reduce_color_space:
            img = reduce_color_space(img, fast=True)

        if options.grayscale:
            img = img.grayscale()

        # normalize to width
        img = img.resize(NORMALIZED_WIDTH, int(img.height*1.0*NORMALIZED_WIDTH/img.width))
        img.save(path.join(model_path, image_file))

def select_models(spec, options):
    model_file, fgvc_data_dir, output_dir = spec

    print 'Emptying %s ...' % output_dir
    os.system('rm -rf %s' % output_dir)
    os.mkdir(output_dir)

    # use set for querying performance
    models = set(m for m in open(model_file) if not m.startswith('#'))

    model_images = defaultdict(list)
    model_image_file = path.join(fgvc_data_dir, 'images_family_trainval.txt')
    with open(model_image_file) as mif:
        for line in mif:
            image_no, model = line.split(' ', 1)
            if model in models:
                model_images[model].append(image_no)

    normalize_model_name = lambda n: n.strip().lower().replace(' ', '-')
    for model, image_nos in model_images.iteritems():
        print 'Processing [%s] images ...' % model.strip()
        model = normalize_model_name(model)

        model_path = path.join(output_dir, model)
        os.mkdir(model_path)

        if options.maximum > 0:
            image_nos = random.sample(image_nos, options.maximum)

        write_images(model_path, fgvc_data_dir, image_nos, options)

        print '--Finished. (total: %d)' % len(image_nos)


if __name__ == '__main__':
    usage = 'Usage: %prog [options] <model_file> <fgvc_data_dir> <output_dir>\n' +\
            'Process images of specified models (or families in FGVC lingo) and organize to SimpleCV convention.\n\n' +\
            '<model_file> is a plain text file with desired models/families each on a single line.\n' +\
            '<fgvc_data_dir> is the data directory of FGVC.\n' +\
            '<output_dir> is the directory where the processed images go.\n' +\
            'WARNING: all contents of <output_dir> will be wiped out so DO NOT put any content there.'

    parser = OptionParser(usage=usage)
    parser.add_option('-m', '--maximum', dest='maximum', type='int', default=0, help='maximum number of samples to use for each model')
    parser.add_option('-c', '--crop', dest='crop', action='store_true', default=False, help='crop images to FGVC boxes')
    parser.add_option('-g', '--grayscale', dest='grayscale', action='store_true', default=False, help='remove color information')
    parser.add_option('-r', '--reduce-color-space', dest='reduce_color_space', action='store_true', default=False, help='reduce color space to specified number of colors')
    options, args = parser.parse_args()

    if len(args) != 3:
        parser.error('Incorrect number of arguments.')

    if not path.exists(args[0]):
        parser.error('<model_file> does not exist.')

    if not path.exists(path.join(args[1], 'images')):
        parser.error('<fgvc_data_dir> does not have images directory.')

    if path.exists(args[2]) and (not path.isdir(args[2])):
        parser.error('<output_dir> exists but is not a directory.')

    select_models(args, options)
