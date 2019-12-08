import cv2
import numpy as np
import argparse
import glob

cap = cv2.VideoCapture(0)
pre_shown = []

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--templates", required=True, help="Path to template image")
args = vars(ap.parse_args())


while True:

    # load video, convert it to grayscale
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # loop over the images to find the template in
    for templatePath in glob.glob(args["templates"] + "/*.png"):

        # load the template, convert it to grayscale
        template_gray = cv2.imread(templatePath, cv2.IMREAD_GRAYSCALE)
        
        w, h = template_gray.shape[::-1]

        # matching to find the template in the frame
        res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
        
        loc = np.where(res >= 0.7)

        for pt in zip(*loc[::-1]):
            
            # this template is not shown again if the template is already shown
            if templatePath not in pre_shown:
                
                # adding new template to pre-shown array
                pre_shown.append(templatePath)
                # draw a bounding box around the detected region
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,0), 3)

    cv2.imshow("frame", frame)

    pre_shown = []
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.DestroyAllWindows()