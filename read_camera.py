import numpy as np
import cv2

MIN_MATCH_COUNT = 30

# Set camera number
cap = cv2.VideoCapture(1)

# Load monitor display image 1: color, 0:grayscale, -1:unchanged
qFrame = cv2.imread('SampleMonitor.jpg',1)
qGrayFrame = cv2.cvtColor(qFrame, cv2.COLOR_BGR2GRAY)

# Create AKAZE detector
detector = cv2.AKAZE_create()

# Get feature points of query frame
qKeypoints, qDescriptors = detector.detectAndCompute(qFrame, None)

#cv2.imshow('query', qFrame)

while(True):
    # Capture frame-by-frame
    ret, tFrame = cap.read()
    
    # Our operations on the frame come here
    tGrayFrame = cv2.cvtColor(tFrame, cv2.COLOR_BGR2GRAY)
    
    # Get feature points of current frame
    tKeypoints, tDescriptors = detector.detectAndCompute(tFrame, None)
    
    # Brute-Force
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Key point matching
    matches = bf.knnMatch(qDescriptors, tDescriptors, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.8 * n.distance:
            goodMatches.append(m)

    if len(goodMatches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ qKeypoints[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts = np.float32([ tKeypoints[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = qGrayFrame.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        tFrame = cv2.polylines(tFrame,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    
        img = cv2.drawMatches(qFrame,qKeypoints,tFrame,tKeypoints,goodMatches,None,**draw_params)
    
        # Transform the train image
        frame_trans = cv2.warpPerspective(tFrame, np.linalg.inv(M), (720,405))
    
        # Display the resulting frame
        cv2.imshow('frame', frame_trans)
        cv2.imshow('org_frame', tFrame)

    else:
        print "Not enough matches are found - %d/%d" % (len(goodMatches),MIN_MATCH_COUNT)
        matchesMask = None


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

