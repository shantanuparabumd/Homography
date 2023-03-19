import cv2
import numpy as np
import matplotlib.pyplot as plt


def homography(camera_corners,world_corners):
    A=[]
    for i in range(camera_corners.shape[0]):
        x,y=camera_corners[i,0],camera_corners[i,1]
        xw,yw=world_corners[i,0],world_corners[i,1]
        A.append([x,y,1,0,0,0,-xw*x,-xw*y,-xw])
        A.append([0,0,0,x,y,1,-yw*x,-yw*y,-yw])
    A=np.array(A)
    eigenvalues, eigenvectors = np.linalg.eig(A.T@A)
    min_eig_idx=np.argmin(eigenvalues)
    smallest_eigen_vector=eigenvectors[:,min_eig_idx]
    H=np.reshape(smallest_eigen_vector,(3,3))
    H=H/H[2,2]
    return H

# Define the callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONUP:
        # Add the clicked point to the list of selected points
        points.append([x, y])

# Load the image
height,width=800,600
img = cv2.imread('goal.jpg')
img = cv2.resize(img, (height, width))
# Create a window to display the image
cv2.namedWindow('image')

# Register the mouse callback function
cv2.setMouseCallback('image', mouse_callback)

# Initialize the list of selected points
points = []

while True:
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, quit the program
    if key == ord("q") or len(points)>=4:
        break
    # Draw the detected corners on the original image
    for corner in points:
        x,y = corner
        cv2.circle(img, (x,y), 5, (0,0,255), -1)
    cv2.imshow('image', img)
    cv2.waitKey(1)
# Close the window
cv2.destroyAllWindows()

# Load the destination image and the source image
dest_image = cv2.imread('goal.jpg')
dest_image = cv2.resize(dest_image, (height, width))
src_image = cv2.imread('M_Logo.png')
width,height,_=dest_image.shape
# scale=0.2
# width=int(width*scale)
# height=int(height*scale)
src_image = cv2.resize(src_image, (height, width))

# Define the source points and destination points
src_pts = np.array([[0, 0], [src_image.shape[1], 0], [src_image.shape[1], src_image.shape[0]], [0, src_image.shape[0]]], dtype=np.float32)
dst_pts = np.array(points, dtype=np.float32)

# Find the homography matrix
H, _ = cv2.findHomography(src_pts, dst_pts)

# Warp the source image using the homography matrix
warped_image = cv2.warpPerspective(src_image, H, (dest_image.shape[1], dest_image.shape[0]))
# Convert the image to grayscale
gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

# Create a binary mask for black color
maskw = cv2.inRange(gray, 0, 10)
# # Create a mask for the warped image
mask = np.zeros_like(dest_image)
cv2.fillConvexPoly(mask, dst_pts.astype(int), (255, 255, 255), cv2.LINE_AA)

# Blend the images 

dest_image[maskw==0]=np.array([0,0,0])

warped_image[maskw==255]=np.array([0,0,0])

blended_image=np.array(dest_image+warped_image*0.9,np.uint8)


cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)

cv2.destroyAllWindows()