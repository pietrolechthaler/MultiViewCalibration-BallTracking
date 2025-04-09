from flask import Flask, render_template, jsonify, request
import homography.homography as homography

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
        {"src": "/static/images/out1-dist.jpg", "label": "out1", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out2-dist.jpg", "label": "out2", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out3-dist.jpg", "label": "out3", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out4-dist.jpg", "label": "out4", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out5-undist.jpg", "label": "out5", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out7-undist.jpg", "label": "out7", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out8-undist.jpg", "label": "out8", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out13-undist.jpg", "label": "out13", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out6-undist.jpg", "label": "out6", "originalWidth": 3840, "originalHeight": 2160},
        {"src": "/static/images/out12-undist.jpg", "label": "out12", "originalWidth": 3840, "originalHeight": 2160}
    ]
    return jsonify(image_list)

"""
Service to get the list of correspondence points
"""
@app.route('/click', methods=['POST'])
def click():

    # Get position of the click
    data = request.json
    x = data.get('x')
    y = data.get('y')

    print(f"> Click: x={x}, y={y}")

    # Trasform the coordinates of the click to the real world
    x_world, y_world = homography.get_world_point(x, y)

    # If point is not inside the court, return empty list of points
    if(x_world is None and y_world is None):
        print("- Click outside the court")
        return jsonify({"status": "failure", "points": []})
    
    print(f"- Coordinates of real world: ({x_world:.3f}, {y_world:.3f})")

    # Get list of correspondence points
    points = homography.getCorrespondences(x_world, y_world)
    print(f"- Points: {points}")
    
    return jsonify({"status": "success", "points": points})

if __name__ == '__main__':
    app.run(debug=True)
