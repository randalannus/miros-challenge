""" REST API for the background removal service implemented using Flask.
"""

from io import BytesIO
import requests
import flask
from flask import request
import flask_cors
from PIL import Image

from bg_service import BGService


app = flask.Flask(__name__)
flask_cors.CORS(app)
bg_service = BGService()

def main():
    app.run("0.0.0.0", 5000)

@app.post("/")
def remove_background():
    if "image" in request.files:
        file = request.files["image"]
        img = Image.open(file)
    elif request.content_type == "application/json":
        img_url = request.get_json(force=True)["url"]
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
    else:
        return "Invalid content type", 400

    foreground = bg_service.remove_background(img)
    return serve_pil_image(foreground)

def serve_pil_image(img):
    img_io = BytesIO()
    img.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return flask.send_file(img_io, mimetype='image/jpeg')

if __name__ == "__main__":
    main()
