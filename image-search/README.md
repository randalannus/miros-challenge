# Image Search

A service for searching products by their images with a text query.

## Running

### 1. From Docker Hub
1. Pull the image from Docker Hub
```docker pull randalannus/miros-challenge:image-search```
2. Run the docker container. The service runs internally on port 5000.
```docker run -p 5001:5000 randalannus/miros-challenge:image-search```
exposes the service on port 5001.

### 2. Building the Docker image
1. Clone the git repo
2. Create a folder in the root directory named `products`.
Populate the `products` folder with folders of JPEG images. Each subfolder should have the name of the product.
This is the same structure as the `sample` folder provided in the task.
3. Run the `remove_backgrounds_script.py` script. The background removal service must be running for this step to work.
Check or edit the `remove_backgrounds_script.py` for the background removal service address.
4. Build the docker image of this repo
```docker build -t image-search:latest .```
5. Run the docker container. The service runs internally on port 5000.
```docker run -p 5001:5000 image-search:latest```
exposes the service on port 5001.

## Using the service
The only exposed endpoint is POST `/`.
The request body should be JSON. For example
```json
{
    "text": "This is the query",
    "n": 4
}
```
The argument `n` signifies the number of results the service should return.