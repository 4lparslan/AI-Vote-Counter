# regions are:
##    l0    l1    l2    ##
##    ##    ##    ##    ##
## r0 ## r1 ## r2 ## r3 ##
##    ##    ##    ##    ##

def analyze_ballot(detections, region_lines):
    votes = [0, 0, 0, 0]
    line_x_list = []
    is_valid = False

    for line in region_lines:
        line_x_list.append(line[0][0])

    for detection in detections:
        seal_x1 = detection[0]  # left side
        seal_x2 = detection[2]  # right side

        if seal_x1 < line_x_list[0] or seal_x2 < line_x_list[0]:
            votes[0] = 1
        if (seal_x1 < line_x_list[1] and seal_x1 > line_x_list[0]) or (
                seal_x2 < line_x_list[1] and seal_x2 > line_x_list[0]):
            votes[1] = 1
        if (seal_x1 < line_x_list[2] and seal_x1 > line_x_list[1]) or (
                seal_x2 < line_x_list[2] and seal_x2 > line_x_list[1]):
            votes[2] = 1
        if seal_x1 > line_x_list[2] or seal_x2 > line_x_list[2]:
            votes[3] = 1

    if votes.count(1) == 1:
        is_valid = True

    return is_valid, votes
