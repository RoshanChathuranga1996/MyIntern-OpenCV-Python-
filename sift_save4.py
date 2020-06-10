import cv2
import numpy as np
from matplotlib import pyplot as plt


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


img1 = cv2.imread('img/jp_gates_original.png')
kp1, des1 = sift_create(img1)

np.savetxt("C:\\Users\\Roshan\\Desktop\\New folder\\Single_des.npy", des1)

img2 = cv2.imread('img/jp_gates_original.png')
kp2, des2 = sift_create(img2)

orgDes = np.loadtxt("C:\\Users\\Roshan\\Desktop\\New folder\\Single_des.npy")
print(orgDes)
print(des2)
#orgDes = orgDes.any()
#des2 = des2.any()

#if orgDes == des2:
 #   print('-------------------')
  #  print("Equal descriptors")

matches = bf_matcher(np.asarray(orgDes, np.float32), np.asarray(des2, np.float32))
matchesGood = good_ratio(matches)

result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matchesGood, None, flags=2)

keypoint_count = 0

if len(kp1) <= len(kp2):
    keypoint_count = len(kp1)

else:
    keypoint_count = len(kp2)

percentage = (len(matchesGood) / keypoint_count) * 100
print('Matching percentage : ' + str(percentage) + '%')

font = cv2.FONT_HERSHEY_PLAIN
txt = 'Matching percentage : ' + str(percentage) + '%'
result = cv2.putText(result, txt, (10, 30), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

plt.imshow(result)
plt.show()
