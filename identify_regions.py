import cv2 as cv
import numpy as np


def checkForLength(scanner, t):
    t = int(t)
    lengths = []
    start_index = -1
    for i in range(len(scanner)):
        if scanner[i] == 255:
            if start_index == -1:
                start_index = i
        else:
            if start_index != -1:
                if i - start_index > t:
                    lengths.append((start_index, i - 1))
                start_index = -1

    if (start_index != -1) and (len(scanner) - start_index > t):
        lengths.append((start_index, len(scanner) - 1))
    return lengths


def getXCoord(start_row):
    longest_sequence = []
    current_sequence = []
    for i, num in enumerate(start_row):
        if num == 255:
            current_sequence.append(i)
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence.copy()
            current_sequence = []
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence.copy()
    start_index = longest_sequence[0]
    end_index = start_index + len(longest_sequence) - 1
    return (start_index + end_index) // 2


def is_recurring(start_p, end_p, lines):
    ref_x = start_p[0]

    for line in lines:
        # start point line[0]
        ## start point x = line[0][0]
        ## start point y = line[0][1]
        # end point line[1]
        ## end point x = line[1][0]
        ## end point y = line[1][1]

        line_x = line[0][0]

        if abs(line_x - ref_x) < 20:
            return True
    return False


def get_region_lines(gray):
    height, width = gray.shape[:2]
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                              cv.THRESH_BINARY, 15, -2)

    # specify the filtering thresholds
    threshW = 20  # Number of columns used per scan
    threshH = height / 2  # Minimum height of line  (Will be calculated at the end of the process)

    output_lines = []

    # scan the image
    for i in range(0, width, threshW):
        roi = bw[0:height, i:i + threshW]

        # create a scanner 1x [img height]  -> this is a indicator for each roi area
        scanner = [[None] for _ in range(height)]

        # detect lines
        for j in range(len(scanner)):
            # check for white pixels
            if np.any(roi[j] == 255):  # we are looking for white pixels
                scanner[j] = 255
            else:
                scanner[j] = 0

        # filter by line lengths
        a = checkForLength(scanner, threshH)

        if len(a) != 0:
            # draw the line
            start_index = a[0][0]
            end_index = a[0][1]

            # get endpoints' x coordinates
            start_x_coord = getXCoord(roi[start_index])
            end_x_coord = getXCoord(roi[end_index])

            # threshold for eliminating border lines
            border_thresh = 60

            # eliminate the border lines. We just need lines in the middle
            if not ((start_x_coord + i <= border_thresh) or (start_x_coord + i >= width - border_thresh) \
                    or (end_x_coord + i <= border_thresh) or (end_x_coord + i >= width - border_thresh)):
                start_point = (i + start_x_coord, start_index)
                end_point = (i + end_x_coord, end_index)

                if is_recurring(start_point, end_point, output_lines) != True:
                    output_lines.append([start_point, end_point])
    return output_lines
