from preprocess import load_and_preprocess_image
from contour_detection import find_document_contour, warp_perspective
import cv2
import os

def save_and_display_result(original, warped):
    # Save the result
    cv2.imwrite("original_image.jpg", original)
    cv2.imwrite("scanned_document.jpg", warped)
    
    # Open the saved images using the default system viewer
    os.system("start original_image.jpg")
    os.system("start scanned_document.jpg")


def main(image_path):
    # Load and preprocess the image
    original, edged = load_and_preprocess_image(image_path)
    
    # Find the document contour
    contour = find_document_contour(edged)
    
    if contour is not None:
        print(f"Contour points: {contour}")
        # Apply perspective transformation
        warped = warp_perspective(original, contour)
        
        # Save and display the result
        save_and_display_result(original, warped)
    else:
        print("Document contour not found.")

if __name__ == "__main__":
    main("images/sample.jpg")
