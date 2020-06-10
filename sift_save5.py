import cv2
from matplotlib import pyplot as plt
from imutils import paths
import time
import pickle
import numpy as np


def sift_create(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des


def bf_matcher(d1, d2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)
    return matches


def good_ratio(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
    return good


basePaths = list(paths.list_images('C:\\Users\\Roshan\\Desktop\\Intern\\Opencv\\101_ObjectCategories\\camera'))
samplePaths = list(paths.list_images('C:\\Users\\Roshan\\Desktop\\Intern\\Opencv\\101_ObjectCategories\\camera_new'))

start = time.time()
matchedList = []
saveBase = []
loadBase = []
temp = []
tmp = []

for (i, basePath) in enumerate(basePaths):

    imageBase = cv2.imread(basePath)
    kp1, des1 = sift_create(imageBase)

    for point1 in kp1:
        temp = (point1.pt, point1.size, point1.angle, point1.response, point1.octave, point1.class_id, saveBase.append(temp))

    with open("C:\\Users\\Roshan\\Desktop\\New folder\\Keypoints.txt", "rb+") as file:
        file.write(bytes(pickle.dumps(saveBase)))

    if len(basePath) == 1:
        np.savetxt("C:\\Users\\Roshan\\Desktop\\New folder\\Descriptor.npy", des1)
    else:
        orgDes = np.loadtxt("C:\\Users\\Roshan\\Desktop\\New folder\\Descriptor.npy")
        newDes = np.append(np.asarray(orgDes, np.float32), np.asarray(des1, np.float32))
        np.savetxt("C:\\Users\\Roshan\\Desktop\\New folder\\Descriptor.npy", newDes)

    print("Processing image {} ".format(len(basePath)))
    print("********************")


print('{INFO} Keypoint & Descriptor files are created.')
print('Sample images processing........')

for (j, samplePath) in enumerate(samplePaths):

    imageSample = cv2.imread(samplePath)
    kp2, des2 = sift_create(imageSample)

    orgDes = np.loadtxt("C:\\Users\\Roshan\\Desktop\\New folder\\Descriptor.npy")

    with open("C:\\Users\\Roshan\\Desktop\\New folder\\Keypoints.txt", "rb+") as pcl:
        orgKp = pickle.load(pcl)

    for point2 in orgKp:
        tmp = cv2.KeyPoint(x=point2[0][0], y=point2[0][1], _size=point2[1], _angle=point2[2], _response=point2[3], _octave=point2[4], _class_id=point2[5])
        loadBase.append(tmp)

    matches = bf_matcher(np.asarray(orgDes, np.float32), np.asarray(des2, np.float32))
    matchesGood = good_ratio(matches)
    count = len(matchesGood)
    print('Number of keypoints : ' + str(count))

    keypoint_count = 0

    if len(kp1) <= len(kp2):
        keypoint_count = len(kp1)

    else:
        keypoint_count = len(kp2)

    percentage = (len(matchesGood) / keypoint_count) * 100
    print('Matching percentage : ' + str(percentage) + '%')
    print('--------' * 7)

    if percentage > 9:

        matchedList.append([samplePath])
        cv2.imshow('testing_img', imageSample)

        font = cv2.FONT_HERSHEY_PLAIN
        txt = 'Matching percentage : ' + str(percentage) + '%'
        result = cv2.drawMatchesKnn(imageBase, kp1, imageSample, kp2, matchesGood, None, flags=2)
        result = cv2.putText(result, txt, (10, 30), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

        plt.imshow(result)
        plt.show()

    else:
        None

print(orgKp)
