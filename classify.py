import os
import json
import time
import random
from glob import glob
from SimpleCV.ImageClass import Image
from SimpleCV.MachineLearning import SVMClassifier
from SimpleCV.Features import HueHistogramFeatureExtractor, EdgeHistogramFeatureExtractor
from feature_extractors import BasicFeatureByGridExtractor

def average(numbers):
    return 1.0 * sum(numbers) / len(numbers)

def run_cross_validation(config):
    print 'Preparing...'
    # configure extractors to use
    extractors = []
    if config.use_hue_histogram:
        extractors.append(HueHistogramFeatureExtractor())
        print '--loaded HueHistogramFeatureExtractor'
    if config.use_edge_histogram:
        extractors.append(EdgeHistogramFeatureExtractor())
        print '--loaded EdgeHistogramFeatureExtractor'
    if config.use_basic_grid:
        extractors.append(BasicFeatureByGridExtractor())
        print '--loaded BasicFeatureByGridExtractor'

    # config
    # TODO
    svm_properties = {
        'KernelType':'RBF', #default is a RBF Kernel
        'SVMType':'NU',     #default is C
        'nu':0.05,          # NU for SVM NU
        'c':None,           #C for SVM C - the slack variable
        'degree':None,      #degree for poly kernels - defaults to 3
        'coef':None,        #coef for Poly/Sigmoid defaults to 0
        'gamma':None,       #kernel param for poly/rbf/sigma - default is 1/#samples
    }

    data_dir = config.data_dir
    # dammit, .DS_Store
    classes = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

    # get images of each class
    image_sets = []
    for cls in classes:
        if config.double_depth:
            images = [Image(img_file) for img_file in glob(os.path.join(data_dir, cls, '*/*.jpg'))]
        else:
            images = [Image(img_file) for img_file in glob(os.path.join(data_dir, cls, '*.jpg'))]
        print '--loaded %d images of %s' % (len(images), cls)
        random.shuffle(images)
        image_sets.append(images)

    print '\nCross Validating...'
    # start k-fold cross validation
    k = config.k
    results = []
    train_accuracy = []
    test_accuracy = []
    for i in range(k):
        print '--round %d/%d' % ((i+1), k)
        # construct new SVM
        svm = SVMClassifier(extractors, svm_properties)
        # generate train sets and test sets
        train_sets, test_sets = [], []

        for image_set in image_sets:
            n = len(image_set)
            split_point = i * (n/k)
            # as the data are already shuffled, operate linearly is fine
            test_set = image_set[split_point:split_point+n/k]
            train_set = image_set[:split_point]
            train_set.extend(image_set[split_point+n/k:])

            train_sets.append(train_set)
            test_sets.append(test_set)

        # gather accuracy
        train_result = svm.train(train_sets, classes, verbose=False)
        train_accuracy.append(train_result[0])
        print '----train accuracy: %.2f%%' % train_result[0]
        test_result = svm.test(test_sets, classes, verbose=False)
        test_accuracy.append(test_result[0])
        print '----test accuracy: %.2f%%' % test_result[0]

        results.append([train_result, test_result])

    print '\nReporting...'
    print '--training accuracy: [%s], average: %.2f%%' % (', '.join('%.2f%%' % ac for ac in train_accuracy), average(train_accuracy))
    print '---testing accuracy: [%s], average: %.2f%%' % (', '.join('%.2f%%' % ac for ac in test_accuracy), average(test_accuracy))

    output = config.output if config.output else 'output-%d' % int(time.time())
    out_file = '%s.json' % output
    with open(out_file, 'wb') as out:
        data = {'config': vars(config), 'results': results}
        json.dump(data, out)

    print '--result of each round is saved to %s' % out_file

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentError

    parser = ArgumentParser(description='Run k-fold cross validation on aircraft classification')
    parser.add_argument('data_dir', metavar='<image data directory>')
    parser.add_argument('output', metavar='[output file name]', nargs='?')
    parser.add_argument('-k', type=int, default=5, help='rounds of cross validation')
    parser.add_argument('-d', '--double-depth', action='store_true', default=False, help='use if the data are in sub directories')
    parser.add_argument('-hue', '--use-hue-histogram', action='store_true', default=False, help='use hue histogram features')
    parser.add_argument('-edge', '--use-edge-histogram', action='store_true', default=False, help='use edge histogram features')
    parser.add_argument('-basic', '--use-basic-grid', action='store_true', default=False, help='use basic grid features')

    args = parser.parse_args()
    print args

    if not (args.use_edge_histogram or args.use_hue_histogram or args.use_basic_grid):
        raise ArgumentError(None, 'should specify at least one feature extractor')

    run_cross_validation(args)
