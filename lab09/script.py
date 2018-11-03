import cv2, numpy as np
import os


import sys
import numpy as np
def get_image_from_path(img_path, flags=0):
    if not os.path.exists(img_path):
        exit(-1)
    else:
        return cv2.imread(img_path, flags=flags)


img = get_image_from_path("lab09_img/face.jpg") # cv2.imread("lab09_img/face.jpg", flags=cv2.IMREAD_GRAYSCALE)
img_colored = cv2.imread("lab09_img/face.jpg", flags=cv2.IMREAD_COLOR)



def face_detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    id_f = 0
    l_data = []
    w_data = []
    if cap.isOpened() == True:
        while cap.isOpened():

            ret, frame = cap.read()
            if ret == True:
                if id_f % 10 == 0:
                    f2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier('./pesi.xml')
                    faces = face_cascade.detectMultiScale(f2)
                    l_data = []
                    w_data = []
                    for (x, y, w, h) in faces:

                        l_data.append(frame[y:y+h, x:x+w])
                        w_data.append((w, h))
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=5)
                else:
                    for i in np.arange(0, len(l_data)):
                        res = cv2.matchTemplate(frame, l_data[i], cv2.TM_SQDIFF)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        frame = cv2.rectangle(frame, (min_loc[0], min_loc[1]), (min_loc[0] + w_data[i][0] , min_loc[1] +
                                                                                w_data[i][1]), color=(0, 0, 255), thickness=5)
            id_f = id_f +1
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break



    cap.release()
    cv2.destroyAllWindows()

def face_detect_img(img_path, classifier="haarcascade_frontalface_default.xml", show=True):

    img_colored = get_image_from_path(img_path, flags=True)
    img_gray = get_image_from_path(img_path, flags=True)
    face_cascade = cv2.CascadeClassifier(classifier)
    faces = face_cascade.detectMultiScale(img_gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show:
        cv2.imshow("faces", img_colored)
        cv2.waitKey(0)

def main():

    Faces = False
    Video = True


    if Faces:
        face_detect_img("lab09_img/face.jpg")
    if Video:
        face_detect_video("lab09_img/Julia_Roberts.avi")

if __name__ == '__main__':
    main()
