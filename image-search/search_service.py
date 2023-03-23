import os
from dataclasses import dataclass
import glob
from typing import TypeVar
import torch
import clip
from PIL import Image
import numpy as np

# Path to directory of all products with unmodified images
PRODUCTS_PATH = "products"
# Path to directory of all products with background removed images
BGLESS_PATH = "bglessProducts"

@dataclass
class Product:
    name: str
    main_image_path: str # Path to the image to represent the product
    images: list[Image.Image]
    bgless_images: list[Image.Image]
    image_tensors: list[torch.Tensor] | None = None


class SearchService:
    """ Search for products by text.
    """
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)
        self.products = self._load_products()
        # tensor of all images of all products
        self._image_tensor = self._concat_image_tensors(self.products)

    def _load_products(self) -> list[Product]:
        products = []
        for directory in os.walk(PRODUCTS_PATH):
            directory_path = directory[0]
            parts = os.path.split(directory_path)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            product_name = parts[1]

            images = self._images_from_dir(directory_path)
            _first_image_name = glob.glob("*.jpg", root_dir=directory_path)[0]
            main_image_path = os.path.join(directory_path, _first_image_name)
            _bgless_directory_path = directory_path.replace(PRODUCTS_PATH, BGLESS_PATH, 1)
            bgless_images = self._images_from_dir(_bgless_directory_path)
            tensors = [self._image_to_tensor(image) for image in bgless_images]
            products.append(Product(
                name=product_name,
                main_image_path=main_image_path,
                images=images,
                bgless_images=bgless_images,
                image_tensors=tensors
            ))

        return products

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        return tensor

    def _images_from_dir(self, path) -> list[Image.Image]:
        files = glob.glob("*.jpg", root_dir=path)
        return [Image.open(os.path.join(path, file)) for file in files]

    def _concat_image_tensors(self, products: list[Product]) -> torch.Tensor:
        tensors = []
        for product in products:
            tensors.extend(product.image_tensors)
        return torch.cat(tensors, dim=0)

    def search(self, quantity: int, query: str) -> list[tuple[np.float32, Product]]:
        """ Find the `quantity` best matching product for the `query`.
        Return a list of tuples ordered by score, the first element is the product and
        the second element is the product's score.
        """
        text_tensor = clip.tokenize([query]).to(self._device)
        # Calculate the similarity scores for each image. Logits are cosine similarity * 100
        # by CLIP documentation.
        with torch.no_grad():
            logits_per_image, logits_per_text = self._model(self._image_tensor, text_tensor)
        # Flatten and normalize scores
        image_scores = logits_per_text.cpu().numpy().reshape(-1) / 100
        product_scores = self._assign_scores_to_products(image_scores)
        best_scorers = self._sort_products_by_score(product_scores)
        return best_scorers[0:quantity]


    def _assign_scores_to_products(self, scores: list[np.float32]) -> list[np.float32]:
        """Finds the average score for each product across all the product's images.
        The results are ordered in the order of products.
        """
        image_counts = [len(product.images) for product in self.products]
        partitioned_scores = partition_list(scores, image_counts)
        average_scores = [np.average(partition) for partition in partitioned_scores]
        return average_scores

    def _sort_products_by_score(self, scores: list[np.float32]):
        pairs = zip(self.products, scores)
        # Sort the pairs based on scores in descending order
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return sorted_pairs

T = TypeVar("T")
def partition_list(input_list: list[T], lengths: list[int]) -> list[list[T]]:
    """Partitions the input_list to sublists of lengths specified in the lengths parameter"""
    sublists = []
    start = 0
    for length in lengths:
        end = start + length
        sublists.append(input_list[start:end])
        start = end
    return sublists
