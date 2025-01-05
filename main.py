import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from numpy.typing import NDArray

# Configure Tesseract command
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_image(image_path: str) -> NDArray:
    """
    Load image and return the greyscale image
    Args:
        image_path: the path to the image

    Returns:
        NumPy array of greyscale image

    """
    image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_image


def process_image(image: NDArray, threshold: int, blur_kernel: int) -> NDArray:
    """
    Process the image by applying normalization, thresholding, and Gaussian blur.

    Args:
        image: Input grayscale image as a NumPy array.
        threshold: Threshold value for binary thresholding.
        blur_kernel: Kernel size for Gaussian blur.

    Returns:
        Processed image as a NumPy array.
    """
    norm = np.zeros((image.shape[0], image.shape[1]))
    # normalize
    normal_image = cv2.normalize(image, norm, 0, 255, cv2.NORM_MINMAX)
    # binary thresh
    thresh_img = cv2.threshold(normal_image, threshold, 255, cv2.THRESH_BINARY)[1]
    # gaussian blur
    blur_kernel = (
        (blur_kernel, blur_kernel)
        if blur_kernel % 2 == 1
        else (blur_kernel + 1, blur_kernel + 1)
    )
    blur_image = cv2.GaussianBlur(thresh_img, blur_kernel, 0)
    return blur_image


def extract_text(image: NDArray) -> str:
    """Extract text from an image"""
    return pytesseract.image_to_string(image)


# UI
st.title("OCR Application")
st.sidebar.title("Image Processing Settings")
st.write("Upload an image to extract text using OCR.")

# upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # greyscale
    grey_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Display original image
    st.image(image, caption="Original Image", use_container_width=True, clamp=True)

    # filter sliders
    threshold = st.sidebar.slider(
        "Threshold", min_value=50, max_value=255, value=100, step=1
    )
    blur_kernel = st.sidebar.slider(
        "Gaussian Blur Kernel Size", min_value=1, max_value=15, value=1, step=2
    )

    processed_image = process_image(grey_image, threshold, blur_kernel)

    # show new image
    st.image(
        processed_image,
        caption="Processed Image",
        use_container_width=True,
        clamp=True,
        channels="GRAY",
    )

    # Extract text and show text
    extracted_text = extract_text(processed_image)

    st.subheader("Extracted Text")
    st.text(extracted_text)

else:
    st.write("Please upload an image to proceed.")
