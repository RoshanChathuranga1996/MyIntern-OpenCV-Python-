import cv2
from matplotlib import pyplot as plt
from imutils import paths
import time
from numpy import savetxt


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
tCount = []

for (i, basePath) in enumerate(basePaths):

    imageBase = cv2.imread(basePath)
    kp1, des1 = sift_create(imageBase)

    for (j, samplePath) in enumerate(samplePaths):

        imageSample = cv2.imread(samplePath)
        kp2, des2 = sift_create(imageSample)

        matches = bf_matcher(des1, des2)
        matchesGood = good_ratio(matches)
        count = len(matchesGood)
        print('Number of keypoints : ' + str(count))
        total = tCount.append([0])

        keypoint_count = 0
        if len(kp1) <= len(kp2):
            keypoint_count = len(kp1)

        else:
            keypoint_count = len(kp2)

        percentage = (len(matchesGood) / keypoint_count) * 100
        print('Matching percentage : ' + str(percentage) + '%')
        print('--------' * 7)

        if percentage > 60:

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


end = time.time()
print("[INFO] Process took {} seconds".format(end - start))
print('Total checkup cycles : ' + str(len(tCount)))
print('--------' * 7)
print('{} images matched than 60%...'.format(len(matchedList)))
print('Paths to matched images : ')
print(matchedList)
