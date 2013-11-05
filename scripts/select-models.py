import os
import random
from os import path
from SimpleCV import Image
from collections import defaultdict
from optparse import OptionParser, OptionValueError

FGVC_IMAGE_SUFFIX = 'jpg'
FGVC_BANNER_HEIGHT = 20

def write_images(model_path, subdir, fgvc_data_dir, images_nos, options):
    subdir_path = path.join(model_path, subdir)
    os.mkdir(subdir_path)

    for image_no in images_nos:
        image_file = '%s.%s' % (image_no, FGVC_IMAGE_SUFFIX)
        img = Image(path.join(fgvc_data_dir, 'images', image_file))

        # remove out banner
        img_cropped = img.crop(0, 0, img.width, img.height-FGVC_BANNER_HEIGHT)
        img_cropped.save(path.join(subdir_path, image_file))

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
        print 'Processing %s images ...' % model.strip()
        model = normalize_model_name(model)

        model_path = path.join(output_dir, model)
        os.mkdir(model_path)

        if options.maximum > 0:
            image_nos = random.sample(image_nos, options.maximum)

        random.shuffle(image_nos)
        test_number = int(len(image_nos) * options.test_portion)

        write_images(model_path, 'test', fgvc_data_dir, image_nos[:test_number], options)
        write_images(model_path, 'train', fgvc_data_dir, image_nos[test_number:], options)

        print '--Finished. (train: %d, test: %d)' % (len(image_nos)-test_number, test_number)


if __name__ == '__main__':
    usage = 'Usage: %prog [options] <model_file> <fgvc_data_dir> <output_dir>\n' +\
            'Process images of specified models (or families in FGVC lingo) and organize to SimpleCV convention.\n\n' +\
            '<model_file> is a plain text file with desired models/families each on a single line.\n' +\
            '<fgvc_data_dir> is the data directory of FGVC.\n' +\
            '<output_dir> is the directory where the processed images go.\n' +\
            'WARNING: all contents of <output_dir> will be wiped out so DO NOT put any content there.'

    parser = OptionParser(usage=usage)
    parser.add_option('-p', '--portion', dest='test_portion', type='float', default=0.2, help='portion of examples used for test, (0, 1)')
    parser.add_option('-m', '--maximum', dest='maximum', type='int', default=0, help='maximum number of samples to use for each model')
    options, args = parser.parse_args()

    if options.test_portion <= 0 or options.test_portion >= 1:
        parser.error('Portion of example should be in (0, 1)')

    if len(args) != 3:
        parser.error('Incorrect number of arguments.')

    if not path.exists(args[0]):
        parser.error('<model_file> does not exist.')

    if not path.exists(path.join(args[1], 'images')):
        parser.error('<fgvc_data_dir> does not have images directory.')

    if path.exists(args[2]) and (not path.isdir(args[2])):
        parser.error('<output_dir> exists but is not a directory.')

    select_models(args, options)
