import cv2

# Function to segment the image and extract contours and edges
def segment_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return thresholded

# Function to extract contours from the thresholded image
def extract_contours(thresholded_image):
    # Find contours
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to perform edge detection using Canny
def detect_edges(image):
    # Convert to grayscale (if not already)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    return edges

if __name__ == "__main__":
    image_path = "green.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Image {image_path} not found.")
    else:
        # Step 1: Segment the image using thresholding
        thresholded = segment_image(image)
        cv2.imwrite("thresholded_image.jpg", thresholded)
        print("Thresholded image saved as thresholded_image.jpg")
        
        # Step 2: Extract contours from the thresholded image
        contours = extract_contours(thresholded)
        contour_image = image.copy()
        
        # Draw contours on a copy of the original image
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite("contours_image.jpg", contour_image)
        print("Contours image saved as contours_image.jpg")

        # Step 3: Perform edge detection using Canny
        edges = detect_edges(image)
        cv2.imwrite("edges_image.jpg", edges)
        print("Edges image saved as edges_image.jpg")

        # Optionally, display the images
        cv2.imshow("Thresholded", thresholded)
        cv2.imshow("Contours", contour_image)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
