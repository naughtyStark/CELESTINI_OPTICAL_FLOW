import numpy as np
import cv2
import sys
import math as m

try: fn = sys.argv[1]
except: fn = 0


help_message = '''
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
'''
print(help_message)
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[int(step/2):h:step, int(step/2):w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def transform(img,img_size,vertices):
    src = np.float32(vertices)#np.float32([[-200,250],[200,170],[600,170],[1000,250]])
    dst = np.float32([ [0, img_size[1] ], [0, 0], [img_size[0], 0] ,[ img_size[0] , img_size[1] ]])
    perspective = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img,perspective,img_size)
    return warped
    
def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    mask = cv2.bitwise_and(img,mask)
    return mask


def main():
    data = np.load('recording.npy') #images stored in a numpy file
    last_Speed = 0
    '''
    LOGIC : to estimate the speed of the car, we look must look at the road from a top-down perspective(eagle's eye view)
    in order to be able to see the road from a top down perspective, I must take a region in the image which I think is the road and
    change it's perspective to top-down by using a transformation. 
    The transformation is done by the function (transform) which takes the source points and the destination points and 
    the orignal image and gives us a warped image that looks like a top down perspective.
    We then compute the optical flow on these warped images and take the maximum flow speed in it to be the speed of the car
    we can also take the average, however, it is possible that a region in the image does not display optical flow at all and therefore
    it would reduce the average even if the true speed was greater.

    '''
    prevgray = prev = data[0][0]#the images fed to the optical flow compute function have to be in grayscale

    v1y = v4y = prev.shape[0] #y coordinates of end points of roi
    v2y = v3y = int(0.45*prev.shape[0])
    v1x = int(-0.5*prev.shape[1])
    v4x = int(1.5*prev.shape[1])
    v2x = int(0.4*prev.shape[1])
    v3x = int(0.6*prev.shape[1])
    TransformX = prev.shape[1] #end points after transformation. 
    TransformY = prev.shape[0] 

    roi_vertices = np.array([[v1x,v1y],[v2x,v2y],[v3x,v3y],[v4x,v4y]])

    processed_img = roi(prev,[roi_vertices]) #bitmask over the roi so that the optical flow ignores the region outside the roi
    prevgray = transform(processed_img,(TransformX,TransformY),[roi_vertices]) #first image (eagle's eye view)

    for i in range(2000): # i have 2000 images
        
        gray = data[i][0] #current image. in a camera based solution this would be replaced by cap.read() 
        processed_img = roi(img,[roi_vertices]) #bitmasking over roi of current image

        gray = transform(processed_img,(TransformX,TransformY),[roi_vertices]) #warped image
        #optimizations like resizing can help speed up this thing a lot, however it would distract the reader from the main
        #concept. right now, optical flow is computed at 10fps, can be cranked up to 60fps very easily
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)#calculate optical flow
        prevgray = gray #current warped image is the previous image for the next iteration
        flow_X,flow_Y = flow[:,:,0],flow[:,:,1] #X and Y components of motion 
        v = np.sqrt( np.add(np.square(flow_X),np.square(flow_Y)) ) #magnitude of motion at each point.
        
        cv2.imshow('flow', draw_flow(gray, flow)) # to see what the algorithm sees 
        speed = np.max(v) #max observed speed 
        trust = abs(speed-last_Speed)/abs(speed+last_Speed) #adhoc state estimation
        speed = (1-trust)*last_Speed + speed*trust
        last_Speed = speed
        print(speed)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows() 			

main()