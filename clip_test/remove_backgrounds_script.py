""" Converts all images in the 'products' directory to background removed versions and stores
them in the 'bglessProducts' directory.

The folder structure for 'bglessProducts' is the same as 'products' and the file names are
also the same.

The background removal service must be running at 'BGREMOVE_SERVICE_URL'.
"""

import os
import io
import glob
from PIL import Image
import requests

PRODUCTS_PATH = "products"
BGLESS_PATH = "bglessProducts"
BGREMOVE_SERVICE_URL = "http://localhost:5000/"

def remove_bg(path) -> Image.Image:
    with open(path, "rb") as f:
        request_files = {'image': ('image.jpg', f, 'image/jpeg')}
        response = requests.post(BGREMOVE_SERVICE_URL, files=request_files)
    image_data = response.content
    return Image.open(io.BytesIO(image_data))


for directory in os.walk(PRODUCTS_PATH):
    # Create the folder for images
    os.makedirs(directory[0].replace(PRODUCTS_PATH, BGLESS_PATH, 1), exist_ok=True)

    files = glob.glob("*.jpg", root_dir=directory[0])
    for filename in files:
        filepath = os.path.join(directory[0], filename)
        image = remove_bg(filepath)
        bgless_path = filepath.replace(PRODUCTS_PATH, BGLESS_PATH, 1)
        print(bgless_path)
        image.save(bgless_path)



