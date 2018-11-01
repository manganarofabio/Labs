from __future__ import print_function
import cv2
import numpy as np

label_idx = 1
idx = 1


def is_foreground(pixel):
    if pixel != 0:
        return True
    return False


def not_labeled(pixel):
    if pixel > 0 and pixel != 255:
        return False
    return True


def ffilter(img):
    image = img.copy()
    label_idx = 1
    queue = []
    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if is_foreground(img[i, j]) and not_labeled(img[i, j]):
                image[i, j] = label_idx
                queue.append((i, j))
                image = queue_solver(image, queue, label_idx)
                label_idx += 1
    return image


def queue_solver(img, queue, lab):

    while True:
        pix = queue[0]
        for qsi in np.arange(-1, 2):
            for qsj in np.arange(-1, 2):
                if qsi != 0 or qsj != 0:
                    if pix[0] + qsi >= 0 and pix[0] + qsi < img.shape[0] and pix[1] + qsj >= 0 and pix[1] + qsj < img.shape[1]:
                        if is_foreground(img[pix[0]+qsi, pix[1]+qsj]) and not_labeled(img[pix[0] + qsi, pix[1]+qsj]):
                            img[pix[0]+qsi, pix[1]+qsj] = lab
                            queue.append((pix[0]+qsi, pix[1]+qsj))

        queue.pop(0)
        if len(queue) == 0:
            break
    return img


def color_labels(img, num_labels):
    img_colorized = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(0, num_labels):
        img_colorized[img[:, :] == i] = (i * np.array([131, 241, 251]))% 256
    return img_colorized



if __name__ == '__main__':

    img = cv2.imread('lab04_img/cvl_acronym.png', cv2.IMREAD_GRAYSCALE)


    img[img[:, :] != 0] = 255

    #cvl image
    if True:
        img = img * -1 +255


    label_idx = 1
    queue = []
    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if is_foreground(img[i, j]) and not_labeled(img[i, j]):
                img[i, j] = label_idx
                queue.append((i, j))
                while True:
                    pix = queue[0]
                    for qsi in np.arange(-1, 2):
                        for qsj in np.arange(-1, 2):
                            if qsi != 0 or qsj != 0:
                                if pix[0] + qsi >= 0 and pix[0] + qsi < img.shape[0] and pix[1] + qsj >= 0 and pix[1] + qsj < img.shape[1]: #se siamo all'interno dell'immagine
                                    if is_foreground(img[pix[0] + qsi, pix[1] + qsj]) and not_labeled(img[pix[0] + qsi, pix[1] + qsj]):
                                        img[pix[0] + qsi, pix[1] + qsj] = label_idx
                                        queue.append((pix[0] + qsi, pix[1] + qsj))

                    queue.pop(0)
                    if len(queue) == 0:
                        break
                label_idx += 1

    ff = color_labels(img, label_idx)

    cv2.imshow('prova', ff.astype(np.uint8))
    cv2.waitKey(0)