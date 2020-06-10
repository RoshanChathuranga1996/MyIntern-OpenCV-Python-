import cv2
from matplotlib import pyplot as plt
from imutils import paths
import time


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


imagePaths = list(paths.list_images('C:\\Users\\Roshan\\Desktop\\Intern\\Opencv\\101_ObjectCategories\\camera'))
matchedList = []
start = time.time()

query = cv2.imread('img/image_0028.jpg')
qkp, qdes = sift_create(query)
print(qdes)

for (i, imagePath) in enumerate(imagePaths):

    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    image = cv2.imread(imagePath)

    kp, des = sift_create(image)
    matches = bf_matcher(qdes, des)
    matchesGood = good_ratio(matches)
    count = len(matchesGood)
    print('Number of keypoints : ' + str(count))

    keypoint_count = 0
    if len(qkp) <= len(kp):
        keypoint_count = len(qkp)

    else:
        keypoint_count = len(kp)

    percentage = (len(matchesGood) / keypoint_count) * 100
    print('Matching percentage : ' + str(percentage) + '%')
    print('--------' * 7)

    if percentage > 60:

        matchedList.append([imagePath])
        cv2.imshow('testing_img', image)

        font = cv2.FONT_HERSHEY_PLAIN
        txt = 'Matching percentage : ' + str(percentage) + '%'
        result = cv2.drawMatchesKnn(query, qkp, image, kp, matchesGood, None, flags=2)
        result = cv2.putText(result, txt, (10, 30), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

        plt.imshow(result)
        plt.show()

    else:
        None

end = time.time()
print("[INFO] Process took {} seconds".format(end - start))
print('--------' * 7)
print('{} images matched than 60%...'.format(len(matchedList)))
print('Paths to matched images : ')
print(matchedList)

