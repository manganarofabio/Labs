#from __future__ import division

import cv2
import numpy as np
import os
import math





def get_image_from_path(img_path, flags=0):
    if not os.path.exists(img_path):
        exit(-1)
    else:
        return cv2.imread(img_path, flags=flags)


def get_kernel_ones(dim=3):


    k = (dim - 1) // 2
    kernel = np.full((dim, dim), 1)

    return kernel.astype(np.float), k

def sobel(img_path, threshold=100, show=True):

    img = get_image_from_path(img_path, flags=False)

    height, width = img.shape

    kernel, k = get_kernel_ones()

    if show:
        cv2.imshow("normal", img)

    edo = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    if show:
        print(edo)
        print(edv)

    imgko = np.zeros((height, width))
    imgkv = np.zeros((height, width))
    img = img.astype(np.float)
    imgko = imgko.astype(np.float)
    imgkv = imgkv.astype(np.float)

    for i in range(k, height - k):
        for j in range(k, width - k):
            imgko[i, j] = 0
            imgkv[i, j] = 0
            for n in range(-k, k + 1):
                for m in range(-k, k + 1):
                    imgko[i, j] = imgko[i, j] + img[i - n, j - m] * edo[k + n, k + m]
                    imgkv[i, j] = imgkv[i, j] + img[i - n, j - m] * edv[k + n, k + m]

    imgf = np.zeros((height, width))
    imgf = imgf.astype(np.float)

    imgf = np.sqrt((imgko ** 2) + (imgkv ** 2))

    th = threshold

    imgf[imgf > th] = 255
    imgf[imgf <= th] = 0


    imgkv = imgkv.astype(np.uint8)
    imgf = imgf.astype(np.uint8)
    imgko = imgko.astype(np.uint8)

    if show:
        cv2.imshow("edge", imgf)
        cv2.imshow("vertical", imgkv)
        cv2.imshow("orizontal", imgko)
        cv2.waitKey(0)

def magnitude(Gx, Gy):
    tmp = (Gx**2 + Gy**2)**0.5
    norm = ((255.*3/8)**2+(255.*3/8)**2)**0.5

    return (tmp/norm)*255


def theta(Gx, Gy):
    return np.arctan2(Gy, Gx)

def sobel_HSV(img_path, show=True):
    img = get_image_from_path(img_path, flags=False)

    height, width = img.shape

    kernel, k = get_kernel_ones()

    if show:
        cv2.imshow("normal", img)

    edo = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)*(1/8.)
    edv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)*(1/8.)

    if show:
        print(edo)
        print(edv)

    imgko = np.zeros((height, width))
    imgkv = np.zeros((height, width))
    img = img.astype(np.float)
    imgko = imgko.astype(np.float)
    imgkv = imgkv.astype(np.float)

    for i in range(k, height - k):
        for j in range(k, width - k):
            imgko[i, j] = 0
            imgkv[i, j] = 0
            for n in range(-k, k + 1):
                for m in range(-k, k + 1):
                    imgko[i, j] = imgko[i, j] + img[i - n, j - m] * edo[k + n, k + m]
                    imgkv[i, j] = imgkv[i, j] + img[i - n, j - m] * edv[k + n, k + m]




    H = theta(imgko, imgkv)
    V = magnitude(imgko, imgkv)
    S = np.ones(V.shape, dtype=np.float64) * 255
    # hsv = np.array([H, S, V])
    H = (180. - 0) / (np.amax(H) - np.amin(H)) * (H - np.amin(H))
    im = np.zeros((H.shape[0], H.shape[1], 3), dtype=np.float64)
    im[:, :, 0] = H
    im[:, :, 1] = S
    im[:, :, 2] = V
    hsv = im
    hsv = hsv.astype(dtype=np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if show:
        cv2.imshow('magnitude', hsv)
        cv2.waitKey(0)


def log(img_path, threshold=60, show=True):

    LoG = get_image_from_path(img_path, 0)
    LoG = LoG.astype(dtype=np.float64)

    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    LoG = cv2.filter2D(LoG, -1, lap)

    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3, 3)))
    th = threshold
    LoG = np.logical_and(LoG < -th, maxLoG > th) * 255

    if show:
        cv2.imshow('log', LoG.astype(np.uint8))
        cv2.waitKey(0)


def canny(img_path, minCanny=100, maxCanny=200, show=True):

    img = get_image_from_path(img_path, 0)
    edges = cv2.Canny(img, minCanny, maxCanny)

    if show:
        cv2.imshow('canny', edges)
        cv2.waitKey(0)





def main():


    Sobel = False
    Sobel_HSV = False
    Canny = True
    Log = True


    if Sobel:
        sobel('lab03_img/modena.png', threshold=100)

    if Sobel_HSV:
        sobel_HSV('lab03_img/modena.png')

    if Canny:
        canny('lab03_img/modena.png')

    if Log:
        log('lab03_img/modena.png')


if __name__ == '__main__':
    main()