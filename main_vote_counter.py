import cv2 as cv
import imutils
import ballot_crop
import identify_regions
import darknet_worker
import vote_count

# darknet parameters #
config_path = "cfg/yolov4-tiny-custom.cfg"
weights_path = "data/best.weights"
data_path = "data/obj.data"
batch_size = 1
thresh = 0.6
# end of darknet parameters #

### feature matching - ballot detection parameters ###
BALLOT_THRESHOLD = 2
is_ballot = 0
match_number = 0
# Initiate SIFT detector
sift = cv.SIFT_create()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
img_base = cv.imread('ballot-bw.png',cv.IMREAD_GRAYSCALE)          # queryImage
kp1, des1 = sift.detectAndCompute(img_base, None)
### end of feature matching - ballot detection parameters ###

# # For webcam input
# ip_address = "192.168.1.##"
# port = 8080
# video_stream_url = f"http://{ip_address}:{port}/video"

cap = cv.VideoCapture("/home/alp/Documents/cutted.mp4")
success, image = cap.read()
image = cv.resize(image, (1280,720))

# ballot counting and capture-only-once parameters #
ballot_number = 0
ballot_capture = True
# end of ballot counting and capture-only-once parameters #

### motion detection parameters ###
start_frame = imutils.resize(image, width=100)
start_frame = cv.cvtColor(start_frame, cv.COLOR_BGR2GRAY)
start_frame = cv.GaussianBlur(start_frame, (21, 21), 0)
### end of motion detection parameters ###

votes = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0  # invalid votes
}

while success:
    image = cv.resize(image, (1280,720))
    im_out = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    im_out = cv.putText(img=im_out, text="sayilan oy =" +str(ballot_number), org=(50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(0, 255, 255), thickness=2)
    im_out = cv.putText(img=im_out, text="erdogan = " + str(votes[str(0)]), org=(50, 85), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(255, 255, 0), thickness=2)
    im_out = cv.putText(img=im_out, text="ince = " + str(votes[str(1)]), org=(50, 120), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(255, 255, 0), thickness=2)
    im_out = cv.putText(img=im_out, text="kilicdaroglu = " + str(votes[str(2)]), org=(50, 155), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(255, 255, 0), thickness=2)
    im_out = cv.putText(img=im_out, text="ogan = " + str(votes[str(3)]), org=(50, 190), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(255, 255, 0), thickness=2)
    im_out = cv.putText(img=im_out, text="gecersiz = " + str(votes[str(4)]), org=(50, 225),
                        fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=(0, 0, 255), thickness=2)

    cv.imshow("resized", im_out)
    cv.waitKey(10)

    ## motion detection ##
    frame = imutils.resize(gray, width=100)
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    difference = cv.absdiff(frame, start_frame)
    threshold = cv.threshold(difference, 25, 255, cv.THRESH_BINARY)[1]
    start_frame = frame
    motion = threshold.sum()
    ## end of motion detection ##

    ## feature matching and ballot detection ##
    kp2, des2 = sift.detectAndCompute(gray, None)
    matches = flann.knnMatch(des1, des2, k=2)
    match_number = 0
    is_ballot = 0
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.3 * n.distance:
            matchesMask[i] = [1, 0]
            match_number += 1
    if match_number >= BALLOT_THRESHOLD:
        is_ballot = 1
    ## end of feature matching and ballot detection ##

    # is ballot changed
    if not is_ballot:
        ballot_capture = True

    # ready for capturing vote ballot?
    if is_ballot and motion < 1 and ballot_capture:

        image = ballot_crop.cropper(image)

        if image is not None:

            # detect seals
            detections = darknet_worker.get_detections(image, config_path, weights_path, data_path, batch_size, thresh)

            # detect regions
            region_lines = identify_regions.get_region_lines(image)
            ballot_number += 1
            ballot_capture = False
            line_counter = 0
            for line in region_lines:
                image = cv.line(image, line[0], line[1], (0, 255, 0), 3)
                line_counter+=1

            for bbox in detections:
                cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            cv.imshow("q", image)
            cv.waitKey(10)

            is_valid , result = vote_count.analyze_ballot(detections, region_lines)

            if is_valid:
                vote_taker = result.index(1)
                votes[str(vote_taker)] += 1
            else:
                votes[str(4)] += 1

    success, image = cap.read()