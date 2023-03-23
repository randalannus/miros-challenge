# Image Search

A service for removing backgrounds from images.

## Running

### 1. From Docker Hub
1. Pull the image from Docker Hub
```docker pull randalannus/miros-challenge:bg-remover```
2. Run the docker container. The service runs internally on port 5000.
```docker run -p 5000:5000 randalannus/miros-challenge:bg-remover```
exposes the service on port 5000.

### 2. Building the Docker image
1. Clone the git repo
2. Build the docker image of this repo
```docker build -t bg-remover:latest .```
3. Run the docker container. The service runs internally on port 5000.
```docker run -p 5000:5000 bg-remover:latest```
exposes the service on port 5000.

## Using the service
The only exposed endpoint is POST `/`.
The request body should be JSON. For example
```json
{
    "url": "https://download1502.mediafire.com/ma87fgwljxfg86es4Ok-J9Vqly0sBiEUAJRGwXYD1H3nUebj1a8rx1zN0COvQ6_xx0_chLg3PypdmOSUG09ijBtueBui/zhiwg19gg78svjb/z--8271559-2.jpg"
}
```
The argument `url` is the path to the image file.