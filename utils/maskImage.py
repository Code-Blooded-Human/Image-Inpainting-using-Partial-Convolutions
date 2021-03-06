import numpy as np
from cv2 import cv2

def maskImage(img, dim = (32,32,3)):
      ## Prepare masking matrix
    mask = np.full(dim, 255, np.uint8) ## White background
    for _ in range(np.random.randint(1, 10)):
    # Get random x locations to start line
        x1, x2 = np.random.randint(1, 32), np.random.randint(1, 32)
        # Get random y locations to start line
        y1, y2 = np.random.randint(1, 32), np.random.randint(1, 32)
        # Get random thickness of the line drawn
        thickness = np.random.randint(1, 3)
        # Draw black line on the white mask
        cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)

    ## Mask the image
    masked_image = img.copy()
    masked_image[mask==0] = 255

    return masked_image, mask