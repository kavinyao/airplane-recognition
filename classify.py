import os
import SimpleCV as cv

extractors = []
#hhfe = cv.HueHistogramFeatureExtractor()
#extractors.append(hhfe)
ehfe = cv.EdgeHistogramFeatureExtractor()
extractors.append(ehfe)

svm_properties = {
    'KernelType':'RBF', #default is a RBF Kernel
    'SVMType':'NU',     #default is C
    'nu':0.05,          # NU for SVM NU
    'c':None,           #C for SVM C - the slack variable
    'degree':None,      #degree for poly kernels - defaults to 3
    'coef':None,        #coef for Poly/Sigmoid defaults to 0
    'gamma':None,       #kernel param for poly/rbf/sigma - default is 1/#samples
}
svm = cv.SVMClassifier(extractors, svm_properties)

data_dir = 'data'
# dammit, .DS_Store
classes = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

train_paths = [os.path.join(data_dir, c, 'train') for c in classes]
test_paths = [os.path.join(data_dir, c, 'test') for c in classes]

print svm.train(train_paths, classes, verbose=True)
print svm.test(test_paths, classes, verbose=True)
