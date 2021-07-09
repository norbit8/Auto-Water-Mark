import cv2 as cv
import os
import glob

alpha = 0.8
failure = -1


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def main():
    images = listdir_nohidden('./images')
    for image in images:
        image_name = image.split('/')[2]
        src1 = cv.imread(cv.samples.findFile(image))
        src2 = cv.imread(cv.samples.findFile('copy_right.jpeg'))
        if src1 is None:
            print("Error loading src1")
            exit(failure)
        elif src2 is None:
            print("Error loading src2")
            exit(failure)
        # Blending Images
        beta = (1.0 - alpha)
        dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
        cv.imwrite('./water_marked_images/' + image_name, dst)


if __name__ == '__main__':
    main()
