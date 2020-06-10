import cv2
from imutils import paths
import time

basePaths = list(paths.list_images('C:\\Users\\Roshan\\Desktop\\Intern\\Opencv\\101_ObjectCategories\\camera'))
samplePaths = list(paths.list_images('C:\\Users\\Roshan\\Desktop\\Intern\\Opencv\\101_ObjectCategories\\camera'))
start = time.time()

for (i, basePath) in enumerate(basePaths):
    imageBase = cv2.imread(basePath)
    hashImg1 = cv2.img_hash_PHash(imageBase)
    print("Base")
    print(hashImg1)

    for (j, samplePath) in enumerate(samplePaths):

        imageSample = cv2.imread(samplePath)
        hashImg2 = cv2.img_hash_PHash(imageSample)
        print("Sample")
        print(hashImg2)

        if hashImg1 == hashImg2:
            print("Equal images")
            cv2.imshow('BaseImg', imageBase)
            cv2.imshow('SampleImg', imageSample)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        else:
            None

end = time.time()
print('Process take {} seconds'.format(end - start))



