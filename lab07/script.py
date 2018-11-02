import numpy as np
import cv2
import matplotlib.pyplot as plt

def hist(img):
    r = np.zeros((256))
    g = np.zeros((256))
    b = np.zeros((256))

    for x in np.nditer(img[:,:,0]):
        r[x] = r[x]+1
    for x in np.nditer(img[:,:,1]):
        g[x] = g[x]+1
    for x in np.nditer(img[:,:,2]):
        b[x] = b[x]+1
    return np.concatenate((r,g,b), axis=0)


def joint_hist(img, nbin=256):
    dbin = 256./nbin
    hist = np.zeros((nbin, nbin, nbin))
    h,w,z = img.shape

    for r in range(h):
        for c in range(w):
            red = img[r,c,0]
            g = img[r,c,1]
            b = img[r,c,2]

            hist[int(red/dbin), int(g/dbin), int(b/dbin)]+= 1
    return hist.astype(np.float32)


img = cv2.imread('lab07_img/m1.jpg', flags=True)

def print_hist(histogram, title='Histogram'):
    plt.bar([i for i in np.arange(0, len(histogram))], histogram, 1)
    plt.title(title)
    plt.show()






def main():

    img = cv2.imread('lab07_img/m1.jpg', flags=True)

    h = hist(img)
    print_hist(h)


    h1 = hist(cv2.imread('lab07_img/mountain.jpg', flags=True))
    print_hist(h1)

    hjoint1 = joint_hist(img)
    hjoint2 = joint_hist(cv2.imread('lab07_img/mountain.jpg', flags=True))


    tmp = np.resize(hjoint1, (-1))
    tmp1 = np.resize(hjoint2, (-1))

    a = cv2.compareHist(tmp, tmp1, method=cv2.HISTCMP_CORREL)
    print(a)




if __name__ == '__main__':
    main()