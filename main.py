import cv2
import numpy as np
import pytesseract
from PIL import Image
from numpy.typing import NDArray

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_image(image_path: str) -> NDArray:
    image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_image


def show_image(image: NDArray, window_name: str) -> None:
    # Show image
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def basic_image():
    # This is just the sticky note image with a simple grey scale filter on it.
    # We can see that it struggles with some of the text.
    # Read the test image
    image_path = "data/test_image.jpg"
    grey_image = load_image(image_path)
    show_image(grey_image, "Basic Image")
    # Run text extraction
    text = pytesseract.image_to_string(grey_image)

    print(f"GREY SCALE\n" f"Extracted Text: {text}")
    # There is too much noise for the text to be read here.


def basic_filtering():
    # We will use filters to extract only the text from the image.

    image_path = "data/test_image.jpg"
    grey_image = load_image(image_path)

    norm = np.zeros((grey_image.shape[0], grey_image.shape[1]))
    # Normalizing image
    normal_image = cv2.normalize(grey_image, norm, 0, 255, cv2.NORM_MINMAX)
    # Binary threshold on image
    thresh_img = cv2.threshold(normal_image, 100, 255, cv2.THRESH_BINARY)[1]
    # GaussianBlur on image
    blur_image = cv2.GaussianBlur(thresh_img, (1, 1), 0)
    show_image(blur_image, "Filter")

    # Run text extraction
    text = pytesseract.image_to_string(blur_image)

    print(f"Filtered\n" f"Extracted Text: {text}")
    # As we can see by the output, the text detection was able to read the image after the filtering.

if __name__ == "__main__":
    basic_image()
    basic_filtering()
