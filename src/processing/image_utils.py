import io
import numpy as np
import cv2
from PIL import Image, ImageFile
import requests
from torchvision import transforms
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 600
IMG_SIZE_left_right = 300
IMG_SIZE_single = 224

def trim(im: Image.Image) -> Image.Image:
    try:
        img_np = np.array(im.convert("RGB"))
    except Exception:
        return im
    percentage = 0.02
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mean_val = np.mean(img_gray[img_gray != 0]) if np.any(img_gray != 0) else 0
    if mean_val == 0:
        im_bin = img_gray > 0
    else:
        im_bin = img_gray > 0.1 * mean_val
    row_sums = np.sum(im_bin, axis=1)
    col_sums = np.sum(im_bin, axis=0)
    rows = np.where(row_sums > img_np.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_np.shape[0] * percentage)[0]
    if not rows.size or not cols.size:
        return im
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    if max_row <= min_row or max_col <= min_col:
        return im
    im_crop_np = img_np[min_row:max_row + 1, min_col:max_col + 1]
    return Image.fromarray(im_crop_np)

def resize_maintain_aspect(image: Image.Image, desired_size: int) -> Image.Image:
    old_size = image.size
    if old_size[0] == 0 or old_size[1] == 0:
        return Image.new("RGB", (desired_size, desired_size))
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    new_size = (max(1, new_size[0]), max(1, new_size[1]))
    try:
        im = image.resize(new_size, Image.LANCZOS)
    except Exception:
        return image
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im

def apply_clahe_color(image: Image.Image) -> Image.Image:
    try:
        img_np = np.array(image.convert("RGB"))
    except Exception:
        return image
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_image_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final_image_np)

def process_image_for_main_model(image: Image.Image) -> Image.Image:
    trimmed_image = trim(image)
    resized_image = resize_maintain_aspect(trimmed_image, IMG_SIZE)
    final_image = apply_clahe_color(resized_image)
    return final_image

def fetch_image_from_url(image_url: str) -> Image.Image:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Timeout while fetching image from URL: {image_url}.")
    except requests.exceptions.TooManyRedirects:
        raise ConnectionError(f"Too many redirects for URL: {image_url}.")
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(f"HTTP error {e.response.status_code} while fetching image: {image_url}.")
    except requests.exceptions.RequestException:
        raise ConnectionError(f"Failed to fetch image from URL: {image_url} due to a network issue.")
    except IOError:
        raise ValueError(f"Failed to open or read image from URL: {image_url}. The file may be corrupt or not a valid image.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while fetching image {image_url}: {str(e)}")

def preprocess_image_for_left_right_model(image: Image.Image, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_left_right + 20, IMG_SIZE_left_right + 20)),
        transforms.CenterCrop(IMG_SIZE_left_right),
        transforms.ToTensor(),
    ])
    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb)
    return image_tensor.unsqueeze(0).to(device)

def preprocess_image_for_fundus_model(image: Image.Image, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_single, IMG_SIZE_single)),
        transforms.ToTensor(),
    ])
    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb)
    return image_tensor.unsqueeze(0).to(device)

def get_image_transform_for_main_model() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])
