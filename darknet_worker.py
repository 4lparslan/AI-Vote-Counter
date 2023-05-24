import cv2 as cv
import darknet

def resize_bbox(detections, out_size, in_size):
    coord = []
    scores = []

    # Scaling the bounding boxes according to the original image resolution.
    for det in detections:
        points = list(det[2])
        conf = det[1]

        xmin, ymin, xmax, ymax = darknet.bbox2points(points)
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        ymin = int(y_scale * ymin)
        ymax = int(y_scale * ymax)
        xmin = int(x_scale * xmin) if int(x_scale * xmin) > 0 else 0
        xmax = int(x_scale * xmax)

        final_points = [xmin, ymin, xmax - xmin, ymax - ymin]
        scores.append(conf)
        coord.append(final_points)

    return coord, scores

def yolo_det(frame, config_file, data_file, batch_size, weights, threshold, network, class_names, class_colors):

    # Preprocessing the input image.
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    image_resized = cv.resize(image_rgb, (width, height))

    # Passing the image to the detector and store the detections
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=threshold)
    darknet.free_image(darknet_image)

    # Resizing predicted bounding box from 416x416 to input image resolution
    out_size = frame.shape[:2]
    in_size = image_resized.shape[:2]
    coord, scores = resize_bbox(detections, out_size, in_size)
    return coord, scores


def get_detections(img, config_file, weights, data_file, batch_size, thresh):
    # Loading darknet network and classes along with the bbox colors.
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )

    # Reading the image and performing YOLOv4 detection.
    bboxes, scores = yolo_det(img, config_file, data_file, batch_size, weights, thresh, network,
                                        class_names, class_colors)

    detection_results = []

    for bbox in bboxes:
        bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]

        # (x1,y1) is top left point and (x2,y2) is bottom right point of the bbox
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        detection_results.append([x1,y1,x2,y2])

    return detection_results