# Marco Bertolazzi
# matr. 0000884790
# marco.bertolazzi3@studio.unibo.it

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, cos, sin, atan, radians

POWDER_DETECTION_THRESHOLD = 100
ELONGATEDNESS_RATIO = 1.5
MEAN, EIGENVECTORS, EIGENVALUES = 0, 1, 2

images = ["./Images/" + image for image in os.listdir("./Images/")]
images.sort()

def enhance_and_binarize(image):
    #image = cv2.medianBlur(image, 3)
    image = cv2.boxFilter(image, -1, (3, 3))
    threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #close hole step
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

# return also connected components and stats to avoid to calc them twice as they will be used also later...
def basic_blob_classifier(blob_mask):
    # coords in an array of [x,y] pairs (formatted according to PCACompute2 function)
    coords = np.array(np.flip(np.where(blob_mask == 0)).T).astype(np.uint8)
    blob_mask_connected_components, stats = cv2.connectedComponents(blob_mask), cv2.PCACompute2(coords, mean=None)

    #look for the number of holes and elongatedness
    holes_count = blob_mask_connected_components[0] - 2
    if holes_count == 0:
        obj_type = "BOLT"
    elif holes_count == 1:
        elongatedness = sqrt(stats[EIGENVALUES][0] / stats[EIGENVALUES][1])
        obj_type = "WASHER" if elongatedness < ELONGATEDNESS_RATIO else "ROD_A"
    elif holes_count == 2:
        obj_type = "ROD_B"
    else:
        obj_type = "ERROR"

    return obj_type, blob_mask_connected_components, stats

def rod_descriptor(blob_mask, blob_mask_connected_components, stats):
    rod = {}

    rod["Type"] = 'A' if blob_mask_connected_components[0] - 2 == 1 else 'B'
    rod["Position"] = np.array(stats[MEAN][0])

    rod["Holes"] = []
    for label in range(2, blob_mask_connected_components[0]):  # skip background and rod, consider only holes!!
        hole_coords = np.where(blob_mask_connected_components[1] == label)
        Hole = {}
        Hole["Centre"] = (np.mean(hole_coords[1]), np.mean(hole_coords[0]))
        Hole["Diameter"] = 2 * sqrt(len(hole_coords[0]) / pi)  # 2*r = sqrt(area/pi)
        rod["Holes"].append(Hole)

    rod["Orientation"] = atan(stats[EIGENVECTORS][0][1]/stats[EIGENVECTORS][0][0])    # sin(theta)/cos(theta), wrt x axis, increase clockwise
    rod["Orientation"] = rod["Orientation"] * 180 / pi     # in integer degrees

    # reorient coordintates according to major axis
    coords = np.flip(np.where(blob_mask == 0)).T
    coords = coords - stats[MEAN][0]                   #move the origin to the barycentre
    coords = np.dot(stats[EIGENVECTORS], coords.T)  #rotate according to major axis

    rod["Length"] = np.max(coords[0]) - np.min(coords[0])
    rod["Width"] = np.max(coords[1]) - np.min(coords[1])

    zero_M_idx = np.where(np.array(coords[0]).astype(int) == 0)
    rod["WidthAtBarycentre"] = np.max(coords[1][zero_M_idx]) - np.min(coords[1][zero_M_idx])

    return rod

def show_rods(image, rods):
    for rod in rods:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #draw barycentre
        cv2.circle(output, (int(rod["Position"][0]), int(rod["Position"][1])), 2, (255, 0, 0), 1)
        #draw orientation
        p1 = (rod["Position"][0] + rod["Length"]/2 * cos(radians(rod["Orientation"])),
              rod["Position"][1] + rod["Length"]/2 * sin(radians(rod["Orientation"])))
        cv2.arrowedLine(output, (rod["Position"][0], rod["Position"][1]), (int(p1[0]), int(p1[1])), (255, 255, 0), 1,
                        cv2.LINE_AA)
        #draw holes
        for hole in rod["Holes"]:
            cv2.circle(output, (int(hole["Centre"][0]), int(hole["Centre"][1])), 2, (255, 0, 0), 1)
            cv2.circle(output, (int(hole["Centre"][0]), int(hole["Centre"][1])), int(hole["Diameter"]/2), (0, 255, 0), 1)

        #draw width at barycentre line
        p1 = (rod["Position"][0] + rod["WidthAtBarycentre"] / 2 * -sin(radians(rod["Orientation"])),
                rod["Position"][1] + rod["WidthAtBarycentre"] / 2 * cos(radians(rod["Orientation"])))
        p2 = (rod["Position"][0] + rod["WidthAtBarycentre"] / 2 * sin(radians(rod["Orientation"])),
                rod["Position"][1] + rod["WidthAtBarycentre"] / 2 * -cos(radians(rod["Orientation"])))
        cv2.line(output, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 1, cv2.LINE_AA)

        print("Type:", rod["Type"])
        print("Position: (%.2f, %.2f)" %(rod["Position"][0],rod["Position"][1]))
        print("Orientation wrt horiz. axis, count-clockwise: %.2f°" %rod["Orientation"])
        print("Length: %.2f" %rod["Length"])
        print("Width: %.2f" %rod["Width"])
        print("Width at barycentre: %.2f" %rod["WidthAtBarycentre"])
        print("Holes:")
        for hole in rod["Holes"]:
            print("\tCentre: (%.2f, %.2f), diameter: %.2f" %(hole["Centre"][0], hole["Centre"][1], hole["Diameter"]))
        print()
        plt.imshow(output)
        plt.show()

for image in images:
    print("Image:", image)
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    binarized = enhance_and_binarize(image)

    n_labels, labelled_image = cv2.connectedComponents(binarized)

    rods = []
    for label in range(1, n_labels):
        blob_area = np.count_nonzero(labelled_image == label)

        if blob_area >= POWDER_DETECTION_THRESHOLD:
            blob_mask = np.full_like(labelled_image, 255, dtype=np.uint8)
            blob_mask[labelled_image == label] = 0
            obj_type, blob_mask_connected_components, stats = basic_blob_classifier(blob_mask)
            if obj_type == "ROD_A" or obj_type == "ROD_B":
                rod = rod_descriptor(blob_mask, blob_mask_connected_components, stats)
                rods.append(rod)
            elif obj_type == "ERROR":
                print(obj_type)
    show_rods(image, rods)
    # end for label in range(1, n_labels):
# end for image in images: