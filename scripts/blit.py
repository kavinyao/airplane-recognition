import os
import sys
import glob
import SimpleCV as cv

if __name__ == '__main__':
    image_dir = sys.argv[1]
    output = sys.argv[2]

    images = [cv.Image(img_file) for img_file in glob.glob(os.path.join(image_dir, '*.jpg'))]
    min_width = min(img.width for img in images)
    min_height = min(img.height for img in images)

    images = [img.resize(min_width, min_height) for img in images]

    a = 1.0 / len(images)
    blended = images[0]
    n = len(images)
    for i in range(1, n):
        blended = blended.blit(images[i], alpha=a**(1.0/(n-i)))

    blended.save(output)
