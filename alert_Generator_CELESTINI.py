import numpy as np
import cv2
import sys
import math as m
import time

try: fn = sys.argv[1]
except: fn = 0


help_message = '''
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
'''
print(help_message)
def draw_flow(img, flow, step=16): #this is just to visualize the optical flow in the image. This function was not written by me
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

def getAlerts(dim_X, dim_Y, ang, v):#takes the dimensions of the image, the direction of motion at each point and the speed of that point
    mid_Y = int(0.3*dim_Y) #wherever the horizon is in the image. mid_Y is not necessarily at the center of the image.
    roi_up = mid_Y #we can work with information in a small region. As in, the sky is not important to us.
    roi_down = int(2*mid_Y) #we don't want to look at the hood of the car
    roi_left = int(0.2*dim_X) #Not interested in what is happening 20 meters to my left or right
    roi_right = int(0.8*dim_X)

    ref_X = int(0.5*dim_X) # X axis position of camera
    ref_Y = dim_Y #bottom of the image. Y axis position of camera
    count = (roi_down - roi_up)*(roi_right-roi_left) #number of pixels in roi 
    danger = 0 #amount of danger
    '''
    LOGIC : 
    ->An object approaching the car would have a velocity who's direction is towards the bottom center of the image(or in human terms,
    towards the car).
        -> component of velocity of an object towards the car 
            = (speed of point)*cos(direction of movement of the point - direction of line joining that point to the bottom center of the image)
    ->the speed of an object is higher for the same pixel movement if the object is near the horizon, to compensate for that,
      we must multiply it by a correction factor, 
        ->corrected speed = sec^2(angle of declination) * original speed. 
    ->the amount of threat posed by an individual point must be proportional to the component of the velocity towards the car
    ->the total threat in that instant is the normalized sum of the threat due to all points in the region of interest.  
    '''
    for i in range(roi_up, roi_down ): # rows (y's)
        for j in range(roi_left, roi_right): #columns (x's)
            #initial point is the ref point. We're drawing a line from the ref point to the point under consideration.
            app_ang = m.atan2( (ref_Y-i) , ref_X-j ) #angle at which an object would move if it was coming towards ref point(us)(approach angle)
            attack = app_ang - ang[i][j] #difference between approach angle and direction of velocity is the angle of attack
            speed = v[i][j]/m.cos( m.atan2( (i-mid_Y)/mid_Y, 1) ) #corrected speed.
            danger += speed*m.pow(m.cos(attack),6) #threat per pixel. I have rased the cosine to the power 6 in order to ignore angles of attack greater than 60 degrees.
    danger /= count #normalize the danger
    return danger

def main():
    data = np.load('recording.npy') #recorded images stored as a numpy array

    prevgray = prev = data[0][0] # first image
    #resizing first image. downsizing image = faster computation (has no effect on the flow of the images)
    prevgray = cv2.resize(prevgray, (int(prevgray.shape[1]/2),int(prevgray.shape[0]/2)), cv2.INTER_LINEAR)
    
    k = 0 # the danger indicating variable. 
    for i in range(2000): #number of sample images is 2000
        gray = data[i][0] #take image. in a real time scenario this would be ret,gray = cap.read() or something
        now = time.time()#this is to check the amount of time taken by the code to execute. on my i5 2.4GHz processor, it ran at ~30-60fps
        
        gray = cv2.resize(gray, (int(gray.shape[1]/2),int(gray.shape[0]/2)), cv2.INTER_LINEAR) #resizing the images
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0) #calculate flow
        prevgray = gray #last image for next instance is current image
        
        flow_X, flow_Y = flow[:,:,0], flow[:,:,1] #the x and y components of motion.
        flow_X = cv2.resize(flow_X,(int(flow_X.shape[1]/4),int(flow_X.shape[0]/4)), cv2.INTER_LINEAR )#resize images 
        flow_Y = cv2.resize(flow_Y,(int(flow_Y.shape[1]/4),int(flow_Y.shape[0]/4)), cv2.INTER_LINEAR )#to reduce computation

        ang = np.arctan2(flow_X, flow_Y) #the angle of motion at each point in the image
        v = np.sqrt( np.add(np.square(flow_X),np.square(flow_Y)) ) #magnitude of motion at each point.

        #low pass filter to reduce false positives
        k = 0.5*k + 0.5*getAlerts(ang.shape[1], ang.shape[0], ang, v) #danger 

        dt = time.time()-now #time taken

        if(k>2): #if danger is greater than a particular threshold.
            print("ALERT!!!!!!") #give alert
        print("dt = ") 
        print(dt)
        cv2.imshow('flow', draw_flow(gray, flow)) #to see what is happening on the screen

        ch = 0xFF & cv2.waitKey(5) #press ESC key to exit
        if ch == 27:
            break
    cv2.destroyAllWindows() 			

main()