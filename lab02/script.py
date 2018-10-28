#from __future__ import division

import cv2
import numpy as np
import math, os.path

def get_image_from_path(img_path, flags=0):
    if not os.path.exists(img_path):
        exit(-1)
    else:
        return cv2.imread(img_path, flags=flags)
        return cv2.imread(img_path, flags=flags)


def get_kernel_ones(dim=3):


    k = (dim - 1) // 2
    kernel = np.full((dim, dim), 1)

    return kernel.astype(np.float), k


def get_gaussian_kernel(sig=1.4):

    dim_kernel = math.floor(5 * sig)
    k = (dim_kernel - 1) // 2

    kernel = np.full((dim_kernel, dim_kernel), 1.0)
    kernel.astype(np.float)

    k_centr = (dim_kernel) // 2

    for i in range(0, dim_kernel):
        for j in range(0, dim_kernel):
            t = (1 / (2 * math.pi * (sig ** 2)))
            # print('t',t)
            s = -1 * ((((i - k_centr) ** 2) + ((j - k_centr) ** 2)) / (2 * (sig ** 2)))
            # print('s',s)
            e = math.exp(s)
            # print('e', e)
            f = t * e
            # print('f',x, f)

            kernel[i, j] = f

    return kernel, k

def conv_Gaussian(img_path, sig=1.4, show=True):

    img = get_image_from_path(img_path)

    imgk = np.zeros((img.shape[0], img.shape[1]))

    imgk = imgk.astype(np.float)


    kernel, k = get_gaussian_kernel(sig)

    for i in range(k, img.shape[0] - k):
        for j in range(k, img.shape[1] - k):
            for n in range(-k, k + 1):
                for m in range(-k, k + 1):
                    imgk[i, j] = imgk[i, j] + img[i - n, j - m] * kernel[k + n, k + m]

    imgk = imgk.astype(np.uint8)

    if show:
        cv2.imshow("Normal", img)
        cv2.imshow("kernel_gaussian", imgk)
        cv2.waitKey(0)

    return imgk


def median_filter(img_path, k_size=3, show=True):

    img = get_image_from_path(img_path)


    imgk = np.zeros((img.shape[0], img.shape[1]))
    l = []
    img = img.astype(np.float)

    kernel, k = get_kernel_ones(k_size)

    for i in range(k, img.shape[0] - k):
        for j in range(k, img.shape[1] - k):
            imgk[i, j] = 0
            for n in range(-k, k + 1):
                for m in range(-k, k + 1):
                    l.append(img[i - n, j - m])

            l.sort()
            imgk[i, j] = l[(kernel.shape[0] * kernel.shape[0]) // 2]
            l[:] = []

    img = img.astype(np.uint8)
    imgk = imgk.astype(np.uint8)
    if show:
        cv2.imshow("normal", img)
        cv2.imshow("kernel_median", imgk)
        cv2.waitKey(0)

    return imgk

def average_filter(img_path, k_size=3, show=True):
    img = get_image_from_path(img_path)

    imgk = np.zeros((img.shape[0], img.shape[1]))

    img = img.astype(np.float)
    imgk = imgk.astype(np.float)


    kernel, k = get_kernel_ones(k_size)
    kernel = kernel / (kernel.shape[0] * kernel.shape[0])

    for i in range(k, img.shape[0] - k):
        for j in range(k, img.shape[1] - k):
            imgk[i, j] = 0
            for n in range(-k, k + 1):
                for m in range(-k, k + 1):
                    imgk[i, j] = imgk[i, j] + img[i - n, j - m] * kernel[k + n, k + m]

    img = img.astype(np.uint8)
    imgk = imgk.astype(np.uint8)

    if show:
        cv2.imshow("normal", img)
        cv2.imshow("kernel_average", imgk)
        cv2.waitKey(0)

    return imgk
#
# def bilateral_filter(img_path, sigmad, sigmar, k_size=3, show=True):
#
#     img = get_image_from_path(img_path)
#
#     bil = np.zeros(img.shape, dtype=np.float64)
#
#     for i in np.arange(k_size/2, img.shape[0]-int(k_size/2)):
#         for j in np.arange(k_size/2, img.shape[1]-int(k_size/2)):
#             num = np.float64(0)
#             den = np.float64(0)
#             for k in np.arange(-int(k_size/2), k_size/2+1):
#                 for l in np.arange(-int(k_size/2), k_size/2+1):
#                     dk = -(((np.float64(i)-np.float64(i+k))**2 + (np.float64(j)-np.float64(j+l))**2) /
#                            np.float64(2*(sigmad**2)))
#                     rk = -(((np.float(img[i, j])-np.float64(img[i+k, j+l]))**2)/(2*np.float64(sigmar**2)))
#                     w = np.float64(np.exp(dk + rk))
#                     num += np.float64(img[i+k, j+l])*w
#                     den += w
#
#             bil[i, j] = num/den
#
#     bil = bil.astype(np.uint8)
#
#     if show:
#         cv2.imshow("kernel_bilateral", bil)
#         cv2.imshow("normal", img)


def main():

    Gaussian = True
    Median = True
    Average = True
    Bilateral = True


    if Gaussian:
        conv_Gaussian("lab02_img\\lena_noise.png")
    if Median:
        median_filter("lab02_img\\lena_noise.png")
    if Average:
        average_filter("lab02_img\\lena_noise.png")
    # if Bilateral:
    #     bilateral_filter("lab02_img\\lena_noise.png", sigmad=1, sigmar=1)


if __name__ == '__main__':
    main()