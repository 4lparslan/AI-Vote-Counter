import numpy as np
import cv2 as cv

AREA_THRESH = 150000
width_image = 1074
height_image = 600

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > AREA_THRESH:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri , True)
            if area > max_area and len(approx) == 4:
                max_area = area
                biggest = approx
    return biggest, max_area

def putInOrder(points):
    sorted_list = []

    # sort top points
    points.sort(key=lambda point: point[1])
    tops = points[:2]
    tops.sort(key=lambda point: point[0])

    sorted_list.append(tops[0])
    sorted_list.append(tops[1])

    # sort bottom points
    points.sort(key=lambda point: point[1], reverse=True)
    bottoms = points[:2]
    bottoms.sort(key=lambda point: point[0])

    sorted_list.append(bottoms[0])
    sorted_list.append(bottoms[1])

    return sorted_list

def cropper(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # find edges with Canny
    edges = cv.Canny(image, 1, 400, apertureSize=3)

    # find contours
    imgContour = image.copy()
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # find the biggest contour
    biggest, max_area = biggestContour(contours)

    font = cv.FONT_HERSHEY_COMPLEX
    n = biggest.ravel()
    i = 0

    four_coord = []

    for j in n:
        if (i % 2 == 0):
            x = n[i]
            y = n[i + 1]

            c = [int(x), int(y)]
            four_coord.append(c)
        i = i + 1

    if len(four_coord) == 4:
        four_coord = putInOrder(four_coord)

        pts1 = np.float32([four_coord])
        pts2 = np.float32([[0, 0], [width_image, 0], [0, height_image], [width_image, height_image]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv.warpPerspective(imgContour, matrix, (width_image, height_image))

        return imgWarp
    return None
