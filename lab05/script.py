import cv2
import numpy as np
import os


def get_image_from_path(img_path, flags=0):
    if not os.path.exists(img_path):
        exit(-1)
    else:
        return cv2.imread(img_path, flags=flags)

def stiching(show=True):

    img1 = cv2.imread('lab05_img/Foto424.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('lab05_img/Foto425.jpg', cv2.IMREAD_COLOR)

    img1 = cv2.resize(img1, (int(img1.shape[1]*0.25), int(img1.shape[0]*0.25)))
    img2 = cv2.resize(img2, (int(img2.shape[1] * 0.25), int(img2.shape[0] * 0.25)))

    if show:
        cv2.imshow('1', img1)
        cv2.imshow('2', img2)

    h, status = cv2.findHomography(np.array([[308, 303], [83, 435], [423, 472], [87, 362]]),
                                   np.array([[608, 303], [383, 435], [723, 472], [387, 362]]))

    im_dst1 = cv2.warpPerspective(img1, h, (img1.shape[1]*2, img1.shape[0]*2))

    h2, status2 = cv2.findHomography(np.array([[430, 23], [216, 166], [552, 187], [214, 92]]),
                                   np.array([[608, 303], [383, 435], [723, 472], [387, 362]]))

    im_dst2 = cv2.warpPerspective(img2, h2, (img1.shape[1] * 2, img1.shape[0] * 2))

    if show:
        cv2.imshow('transformed 1', im_dst1)
        cv2.imshow('transformed 2', im_dst2)

    imgf = np.zeros((img1.shape[0] * 2, img1.shape[1] * 2, 3), dtype=np.uint8)

    imgf[im_dst1[:, :] != [0, 0, 0]] = im_dst1[im_dst1[:, :] != [0, 0, 0]]
    imgf[imgf[:, :] == [0, 0, 0]] = im_dst2[imgf[:, :] == [0, 0, 0]]

    if show:
        cv2.imshow('finale', imgf)
        cv2.waitKey(0)

def hough_lines(img_path, p=True):

    img = get_image_from_path(img_path, True)

    edges = cv2.Canny(img, 100, 150, apertureSize=3)

    cv2.imshow('canny', edges)

    if not p:
        lines = cv2.HoughLines(edges, 1, np.pi / 360, 50)

        if lines is not None:
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    else:

        lines =cv2.HoughLinesP(edges, rho=1, theta=np.pi / 90, threshold=40, minLineLength=20, maxLineGap=50)
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)




    cv2.imshow('img', img)

    cv2.waitKey(0)

def main():

    Stiching = True
    Hough = False

    if Stiching:
        stiching()

    if Hough:
        hough_lines('lab05_img/1.bmp')

if __name__ == '__main__':

    main()