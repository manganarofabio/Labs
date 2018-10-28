from __future__ import division, print_function


__author__ = '190728'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path

#flags == 1 == bgr
def get_image_from_path(img_path, flags=0):
    if not os.path.exists(img_path):
        exit(-1)
    else:
        return cv2.imread(img_path, flags=flags)
        return cv2.imread(img_path, flags=flags)


def get_channels(img):
    if img.ndim == 3 and img.shape[2] > 1:
        return img.shape[2]
    return 1


def resize(img):
    return np.expand_dims(img, 2)

def get_histogram(img_path):

    img = get_image_from_path(img_path)

    # print(img)

    height, width = img.shape
    cv2.imshow("img", img)
    cv2.waitKey(0)

    h = np.zeros((256))

    for x in np.nditer(img):
        h[x] = h[x] + 1

    return h, img

def show_histograms(img_path, normalize=True, reduce=False, streched=False):

    img = get_image_from_path(img_path=img_path)
    # print(img)

    height, width = img.shape
    cv2.imshow("img", img)
    cv2.waitKey(0)

    h = np.zeros((256))

    for x in np.nditer(img):
        h[x] = h[x]+1


    plt.bar(range(256), h, 1)
    plt.title("histogram")
    plt.show()

    if normalize:

        h1 = h/(height*width)

        plt.bar(range(256), h1, 1)
        plt.title("normalized_histogram")
        plt.show()

    if reduce:
        # riduciamo numero di bin a 16
        h16 = np.zeros((16))
        for x in np.nditer(img):
            h16[int((x/256)*16)] = h16[int((x/256)*16)]+1

        plt.bar(range(16), h16, 1)
        plt.title("reduced_histogram")
        plt.show()

    if streched:
        # for x in np.nditer(img):
        #     h[x] = h[x]+1
        # for i, val in (enumerate(h)):
        #     print(i, val)
        #
        # plt.bar(range(256), h)
        # plt.show()

        min = 50
        max = 55
        nmin = 0
        nmax = 255

        h1 = (h - min)*((nmax - nmin)/(max - min))

        plt.bar(range(256), h1, 1)
        plt.title("streched_histogram")
        plt.show()


def get_negative(img_path, show=True):

    img = get_image_from_path(img_path)

    img = img.astype(np.float)
    img = img*-1 + 255
    img[img < 0] = 0
    img[img > 255] = 255

    img = img.astype(np.uint8)
    if show:
        cv2.imshow('negative',img)
        cv2.waitKey(0)
    return img

def get_luminance(img_path, show=True):

    img = get_image_from_path(img_path)
    img = img.astype(np.float)

    img = img*1.5
    img[img < 0] = 0
    img[img > 255] = 255

    img = img.astype(np.uint8)

    if show:
        cv2.imshow('img',img)
        cv2.waitKey(0)

    return img

def get_blend(img1_path, img2_path, alpha,  show=True):



    img1 = get_image_from_path(img1_path, flags=1)
    img2 = get_image_from_path(img2_path, flags=1)

    img1 = img1.astype(np.float)
    img2 = img2.astype(np.float)


    img3 = (1 - alpha)*img1 + alpha*img2

    img3 = img3.astype(np.uint8)

    if show:
        cv2.imshow('img3', img3)
        cv2.waitKey(0)

    # img4 = cv2.imread('mountain.jpg')
    #
    # img1 = img1.astype(np.uint8)
    #
    # img4 = cv2.resize(img4, (img1.shape[1], img1.shape[0]))
    #
    # img1 = img1.astype(np.float)
    # img4 = img4.astype(np.float)
    #
    # a = 0.3
    #
    # img5 = (1 - a)*img1 + a*img4
    #
    # img5 = img5.astype(np.uint8)
    #
    # cv2.imshow('img5', img5)
    # cv2.waitKey(0)



def get_Otsu(img_path, show=True):

    if not os.path.exists(img_path):
        return -1


    h, img = get_histogram(img_path)


    # get normalized hist
    hn = h / (img.shape[0] * img.shape[1])

    # scego una soglia tra 0 e 255

    tmin = 1

    t = 1
    w1 = 0

    m1 = 0
    m2 = 0
    w2 = 0
    s = 0

    w1 = np.sum(hn[0:t + 1])
    print(w1)

    w2 = np.sum(hn[t + 1:256])
    print(w2)

    print('tot', w1 + w2)

    m1 = (np.sum(hn[0:t + 1] * h[0:t + 1])) / w1

    m2 = (np.sum(hn[t + 1:256] * h[t + 1:256])) / w2

    smax = (w1 * w2) * ((m1 - m2) * (m1 - m2))
    tmax = t

    for it in range(2, 255):

        w1 = np.sum(hn[0:it + 1])
        print('w1 ', w1)

        w2 = np.sum(hn[it + 1:256])
        print('w2 ', w2)

        m1 = (np.sum(hn[0:it + 1] * range(0, it + 1))) / w1

        m2 = (np.sum(hn[it + 1:256] * range(it + 1, 256))) / w2

        s = (w1 * w2) * ((m1 - m2) * (m1 - m2))
        # print(it, s)

        if s > smax:
            tmax = it
            smax = s

    print('tmax ', tmax, smax)

    imgO = img

    imgO[img >= tmax] = 255
    imgO[img < tmax] = 0

    img = img.astype(np.uint8)
    imgO = imgO.astype(np.uint8)

    p = np.zeros((256))
    for x in np.nditer(imgO):
        p[x] = p[x] + 1
    # print(p)

    plt.bar(range(256), p)
    plt.show()

    ret2, th2 = cv2.threshold(img, 0, 255, type=cv2.THRESH_OTSU)
    if show:
        cv2.imshow('normal', img)
        cv2.imshow('otsu', imgO)
        cv2.imshow('otsuR', th2)

        cv2.waitKey(0)
    return imgO, th2


def main():

    histograms = False
    Otsu = False
    Negative = False
    Luminance = False
    Blend = True

    if histograms:
        show_histograms("lab01_img\\sunflower.jpg", reduce=True, streched=True)
    if Otsu:
        get_Otsu("lab01_img\\camera.png")
    if Negative:
        get_negative("lab01_img\\sunflower.jpg")
    if Luminance:
        get_luminance("lab01_img\\house.jpg")
    if Blend:
        get_blend("lab01_img\\background.jpg", "lab01_img\\foreground.jpg", alpha=0.5)




if __name__ == '__main__':
    main()


