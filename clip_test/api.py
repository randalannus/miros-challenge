import flask
from flask import request
import flask_cors

from search_service import SearchService


app = flask.Flask(__name__)
flask_cors.CORS(app)

search_service = SearchService()

def main():
    # For debugging purposes
    app.run("0.0.0.0", 5001)

@app.post("/")
def image_search():
    data = request.get_json(force=True)
    text = data["text"]
    n = data["n"]

    results = search_service.search(n, text)
    formatted_results = []
    for result in results:
        formatted_results.append({
            "name": result[0].name,
            "url": result[0].main_image_path,
            "score": float(result[1])
        })

    return formatted_results

if __name__ == "__main__":
    main()
