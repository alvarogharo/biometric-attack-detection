import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def getumbralizationinfo(path, image_names, showResults=False):
    erosion_iterations = 4
    dilation_iterations = 1
    maximum = []
    averages = []
    thickness = []
    iterations = len(image_names)
    count = 0
    for img_name in image_names:
        image = cv2.imread(path + img_name)

        dim = tuple((np.array([image.shape[1], image.shape[0]]) / 4).astype(np.int))
        image = cv2.resize(image, dim)

        imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thrash = cv2.threshold(imgGrey, 215, 240, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)

        erosion = cv2.erode(thrash, kernel, iterations=erosion_iterations)
        dilation = cv2.dilate(erosion, kernel, iterations=dilation_iterations)

        hist = np.flip(np.sum(dilation, axis=1), axis=0)

        maximum.append(np.max(hist))
        avg = np.average(hist)
        averages.append(avg)
        thickness.append(np.sum(hist > avg))

        if showResults == True:
            plt.barh(np.arange(0, 1000, 1), hist)
            plt.show()

            cv2.imshow(img_name, dilation)
            cv2.waitKey()
            cv2.destroyWindow(img_name)
        count += 1
        print(str(round(count/iterations * 100)) + "%")
    thickness_np = np.array(thickness)
    return [np.average(maximum), np.average(averages), np.average(thickness_np[np.nonzero(thickness_np)])]


user_path = "./data/COLOR/USER/"
attack_path = "./data/COLOR/attack_01/"

user_image_names = os.listdir(user_path)
attack_image_names = os.listdir(attack_path)

data_user = getumbralizationinfo(user_path, user_image_names)
data_attack = getumbralizationinfo(attack_path, attack_image_names)
print()
print("[avg max, avg value, avg thickness]")
print("Users info:", data_user)
print("Attacks info:", data_attack)

