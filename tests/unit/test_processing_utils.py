import pytest
from PIL import Image
import numpy as np
from io import BytesIO

# Adjust import path based on how tests will be run (e.g., if src is in PYTHONPATH)
# Assuming tests are run from the root directory and src is discoverable.
from src.processing.image_utils import resize_maintain_aspect, trim

# Fixture to create a simple dummy image
@pytest.fixture
def dummy_image():
    # Create a simple 100x50 black image with a white 20x10 rectangle inside
    # to test trimming and resizing.
    img = Image.new("RGB", (100, 50), "black")
    pixels = img.load()
    for i in range(40, 60): # x-coordinates for white box
        for j in range(20, 30): # y-coordinates for white box
            pixels[i, j] = (255, 255, 255)
    return img

@pytest.fixture
def non_square_image():
    # Create a 150x50 image
    img = Image.new("RGB", (150, 50), "blue")
    return img

def test_resize_maintain_aspect_square_output(non_square_image):
    desired_size = 100
    resized_img = resize_maintain_aspect(non_square_image, desired_size)
    assert resized_img.size == (desired_size, desired_size)

def test_resize_maintain_aspect_larger_to_smaller(dummy_image):
    desired_size = 30 # Smaller than original 100x50
    resized_img = resize_maintain_aspect(dummy_image, desired_size)
    assert resized_img.size == (desired_size, desired_size)
    # Max dimension of original (100) should now be scaled to desired_size (30)
    # Original ratio 100/50 = 2. New max dim is 30.
    # So, if width was max, new width is 30, new height is 30 * (50/100) = 15.
    # Check content if possible, e.g. that it's not all black after resize (harder to check exact content)

def test_resize_maintain_aspect_smaller_to_larger(dummy_image):
    desired_size = 200 # Larger than original 100x50
    resized_img = resize_maintain_aspect(dummy_image, desired_size)
    assert resized_img.size == (desired_size, desired_size)

def test_trim_simple_image(dummy_image):
    # The dummy_image has a white box from (40,20) to (59,29)
    # Trimmed image should be close to these dimensions.
    # Note: trim's percentage calculation might affect exact crop.
    trimmed_img = trim(dummy_image)
    # Expected size is roughly 20x10 (the white box)
    # Allow some tolerance due to trim's percentage logic and potential small artifacts
    assert 15 < trimmed_img.width < 25  # Original white box width is 20
    assert 8 < trimmed_img.height < 15 # Original white box height is 10

    # Verify that the trimmed image is not all black (i.e., it cropped something meaningful)
    img_array = np.array(trimmed_img)
    assert np.sum(img_array) > 0 # Check if there are any non-black pixels

def test_trim_empty_image():
    empty_img = Image.new("RGB", (50, 50), "black") # All black
    trimmed_img = trim(empty_img)
    # Trim should ideally return the original image or an empty version of it
    # if it can't find any content to trim to.
    # Current trim returns original if no rows/cols found.
    assert trimmed_img.size == (50, 50)

    # Verify it's still all black
    img_array = np.array(trimmed_img)
    assert np.sum(img_array) == 0


# It's good to also have a pytest.ini or pyproject.toml to configure pytest,
# especially for path discovery (PYTHONPATH). For now, this assumes src is on path.
