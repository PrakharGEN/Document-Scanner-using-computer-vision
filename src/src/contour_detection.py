import cv2
import numpy as np

def find_document_contour(edged):
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Loop through contours to find the one with 4 points
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            return approx
    return None

def order_points(pts):
    # Convert the points to a 4x2 array
    pts = pts.reshape(4, 2)
    
    # Initialize an array for the ordered points
    ordered_pts = np.zeros((4, 2), dtype="float32")
    
    # Compute the sum and difference of the points
    s = np.sum(pts, axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left point has the smallest sum
    ordered_pts[0] = pts[np.argmin(s)]
    # Bottom-right point has the largest sum
    ordered_pts[2] = pts[np.argmax(s)]
    # Top-right point has the smallest difference
    ordered_pts[1] = pts[np.argmin(diff)]
    # Bottom-left point has the largest difference
    ordered_pts[3] = pts[np.argmax(diff)]
    
    return ordered_pts

def warp_perspective(image, contour):
    # Ensure the contour is in the expected format
    if contour is None:
        raise ValueError("Contour cannot be None")
    if contour.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {contour.shape[0]} points")
    
    # Reorder the contour points
    rect = order_points(contour)
    (tl, tr, br, bl) = rect
    
    # Compute the width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define the destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Perform the perspective transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped
