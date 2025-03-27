from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

"""
Service to get the list of images
"""
@app.route('/images')
def images():
    image_list = [
        {"src": "/static/images/out1-undist.jpg", "label": "out1"},
        {"src": "/static/images/out2-undist.jpg", "label": "out2"},
        {"src": "/static/images/out3-undist.jpg", "label": "out3"},
        {"src": "/static/images/out4-undist.jpg", "label": "out4"},
        {"src": "/static/images/out5-undist.jpg", "label": "out5"},
        {"src": "/static/images/out7-undist.jpg", "label": "out7"},
        {"src": "/static/images/out8-undist.jpg", "label": "out8"},
        {"src": "/static/images/out13-undist.jpg", "label": "out13"},
        {"src": "/static/images/out6-undist.jpg", "label": "out6"},
        {"src": "/static/images/out12-undist.jpg", "label": "out12"}
    ]
    return jsonify(image_list)

"""
Service to get the list of correspondence points
"""
@app.route('/click', methods=['POST'])
def click():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    label = data.get('label') 

    print(f"Click: x={x}, y={y}, image={label}")

    # Correspondence points
    points = {}
    for img in ["out1", "out2", "out3", "out4", "out5", "out7", "out8", "out13", "out6", "out12"]:

        if img == label:
            continue

        # FIXME: this is a simulate behavior (random points)
        points[img] = {
            "x": x + random.randint(-10, 10),
            "y": y + random.randint(-10, 10),
        }

    return jsonify({"status": "success", "points": points})

if __name__ == '__main__':
    app.run(debug=True)
