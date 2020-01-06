import random
import numpy as np
import cv2
import os

def getumbralizedinfo(image, lower_bound, upper_bound, erosion_iterations, dilation_iterations):
    imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thrash = cv2.threshold(imgGrey, lower_bound, upper_bound, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)

    erosion = cv2.erode(thrash, kernel, iterations=erosion_iterations)
    dilation = cv2.dilate(erosion, kernel, iterations=dilation_iterations)

    hist = np.flip(np.sum(dilation, axis=1), axis=0)

    maximum = np.max(hist)
    avg = np.average(hist)
    thickness = np.sum(hist > avg)

    return [maximum, avg, thickness]

def checkresultsandprint(results, false_thickness_threshold):
    ret = 0
    if (results[2] > 0) & (results[2] < false_thickness_threshold):
        print("Attack detected! Printed image border found")
        ret = 1
    else:
        print("Real person detected!")
        ret = 0
    return ret

def detectfaceonIR(path, user_name, faceCascade):
    image = cv2.imread(path + user_name)
    rects_face = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
    ret = 0
    if len(rects_face) == 0:
        print("Attack detected! No person found on IR image")
        ret = 1
    else:
        print("Real person detected!")
        ret = 0
    return ret

def detectattack(image, ir_path, secure=False, mode_ir=False, show_results=False):
    false_bbox_min_size = 291
    false_bbox_max_size = 321
    erosion_iterations = 4
    dilation_iterations = 1
    false_thickness_threshold = 200
    lower_bound = 215
    upper_bound = 240

    dim = tuple((np.array([image.shape[1], image.shape[0]]) / 4).astype(np.int))
    image = cv2.resize(image, dim)

    faceCascade = cv2.CascadeClassifier("./classifiers/haarcascade_frontalface_default.xml")
    rects_face = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
    bbox_sizes = []
    for r in rects_face:
        bbox_sizes.append(r[3])

    ret = 0
    if len(bbox_sizes) == 0:
        print("Attack detected! Any real face found")
        ret = 1
    else:
        bbox_size_avg = np.average(np.array(bbox_sizes))
        if bbox_size_avg < false_bbox_min_size:
            if secure:
                if mode_ir:
                    ret = detectfaceonIR(ir_path, user_name, faceCascade)
                else:
                    results = getumbralizedinfo(image, lower_bound, upper_bound, erosion_iterations, dilation_iterations)
                    ret = checkresultsandprint(results, false_thickness_threshold)
            else:
                print("Attack detected! Face is too small. Probably an image")
                ret = 1
        elif (bbox_size_avg >= false_bbox_min_size) & (bbox_size_avg <= false_bbox_max_size):
            if mode_ir:
                ret = detectfaceonIR(ir_path, user_name, faceCascade)
            else:
                results = getumbralizedinfo(image, lower_bound, upper_bound, erosion_iterations, dilation_iterations)
                ret = checkresultsandprint(results, false_thickness_threshold)
        elif bbox_size_avg > false_bbox_max_size:
            if secure:
                if mode_ir:
                    ret = detectfaceonIR(ir_path, user_name, faceCascade)
                else:
                    results = getumbralizedinfo(image, lower_bound, upper_bound, erosion_iterations, dilation_iterations)
                    ret = checkresultsandprint(results, false_thickness_threshold)
            else:
                print("Real person detected!")
                ret = 0

    if show_results:
        for r in rects_face:
            cv2.rectangle(image, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)

        cv2.imshow("Result", image)
        cv2.waitKey()
    return ret

users_path = "./data/COLOR/USER/"
attacks_path = "./data/COLOR/attack_01/"
users_ir_path = "./data/IR/users/"
attacks_ir_path = "./data/IR/attack_01/"

user_name = "USER_007.JPG"

secure = True
mode_ir = True
show_results = False

user_image_names = os.listdir(users_path)

groundtruth = []
results = []
for image_name in user_image_names:
    rand = random.getrandbits(1)
    groundtruth.append(rand)
    ret = 0
    if rand == 1:
        image = cv2.imread(attacks_path + image_name)
        ir_path = attacks_ir_path
    else:
        image = cv2.imread(users_path + image_name)
        ir_path = users_ir_path
    ret = detectattack(image, ir_path, secure=secure, mode_ir=mode_ir, show_results=show_results)
    results.append(ret)

print("APCER:", str(1-(sum(results)/sum(groundtruth))) + "%")