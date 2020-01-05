import numpy as np
import cv2
import os


def getbboxesinfo(path, image_names, showResults=False):
    faceCascade = cv2.CascadeClassifier("./classifiers/haarcascade_frontalface_default.xml")
    widths = []
    heights = []
    iterations = len(image_names)
    count = 0
    for img_name in image_names:
        image = cv2.imread(path + img_name)

        dim = tuple((np.array([image.shape[1], image.shape[0]]) / 4).astype(np.int))
        image = cv2.resize(image, dim)

        rects_face = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

        if img_name == "USUARIO_058.JPG":
            rects_face = rects_face[0:1]
        elif img_name == "USUARIO_128.JPG":
            rects_face = rects_face[1:2]
        elif img_name == "spoof_058.JPG":
            rects_face = rects_face[2:3]
        elif img_name == "spoof_090.JPG":
            rects_face = rects_face[0:1]

        for r in rects_face:
            widths.append(r[2])
            heights.append(r[3])
            cv2.rectangle(image, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)

        if showResults == True:
            cv2.imshow(img_name, image)
            cv2.waitKey()
            cv2.destroyWindow(img_name)
        count += 1
        print(str(round(count/iterations * 100)) + "%")

    nd_w = np.array(widths)
    nd_h = np.array(heights)

    w = [np.average(nd_w), np.max(nd_w), np.min(nd_w)]
    h = [np.average(nd_h), np.max(nd_h), np.min(nd_h)]

    return w, h


user_path = "./data/COLOR/USER/"
attack_path = "./data/COLOR/attack_01/"

user_image_names = os.listdir(user_path)
attack_image_names = os.listdir(attack_path)

w_users, h_users = getbboxesinfo(user_path, user_image_names)
w_attack, h_attack = getbboxesinfo(attack_path, attack_image_names)

print()
print("[avg, max, min]")
print("Users info:", w_users, h_users)
print("Attacks info:", w_attack, h_attack)



