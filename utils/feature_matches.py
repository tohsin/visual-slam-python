import cv2 as cv
import sys
def find_feature_matches(img_1, img_2):
    orb_detector = cv.ORB_create()

    # find the keypoints with ORB
    kp1, des1 = orb_detector.detectAndCompute(img_1,None)
    kp2, des2 = orb_detector.detectAndCompute(img_2,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    match = bf.match(des1,des2)
    print("number of descriptors", len(des2))
    # compute highest and lowest distance
    min_dist = sys.maxsize
    max_dist = - sys.maxsize

    for i in range(len(match)):
        dist = match[i].distance
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist,  dist)
    print("max distance " ,max_dist )
    print("min distance :", min_dist)
    print("number of matches",len(match))

    matches = []
    for i in range(len(match)):
        if match[i].distance <= max(2 * min_dist, 30.0) :
            matches.append(match[i])
    return matches, kp1, kp2
