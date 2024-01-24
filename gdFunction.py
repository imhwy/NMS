from flask import Flask, request, jsonify
import torch
import cv2
import json

app = Flask(__name__)

def calculate_iou(box1, box2):
    x1, y1, h1, w1 = box1
    x2, y2, h2, w2 = box2

    x1_left, x2_left = (x1 - w1 / 2), (x2 - w2 / 2)
    y1_top, y2_top = (y1 - h1 / 2), (y2 - h2 / 2)
    x1_right, x2_right = (x1 + w1 / 2), (x2 + w2 / 2)
    y1_bottom, y2_bottom = (y1 + h1 / 2), (y2 + h2 / 2)

    intersection_left = max(x1_left, x2_left)
    intersection_top = max(y1_top, y2_top)
    intersection_right = min(x1_right, x2_right)
    intersection_bottom = min(y1_bottom, y2_bottom)

    if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
        intersection_area = 0
    else:
        intersection_area = (intersection_right - intersection_left) * (intersection_bottom - intersection_top)

    area1 = h1 * w1
    area2 = h2 * w2
    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

@app.route('/api/gdFunction', methods=['GET'])
def findBBox():
    try:
        boxes = torch.Tensor(json.loads(request.args.get('boxes')))
        IMAGE_PATH = request.args.get('IMAGE_PATH')
        threshold = float(request.args.get('threshold'))
        # read image
        image = cv2.imread(IMAGE_PATH)
        height, width, _ = image.shape
        boxes = (boxes * torch.Tensor([width, height, width, height])).to(torch.int)
        # remove overlap bounding boxes
        for i in range(len(boxes)):
            for j in range(i+1,len(boxes)):
                iou_value = calculate_iou(boxes[i],boxes[j])
                if iou_value > threshold:
                    boxes = torch.cat((boxes[:j], boxes[j + 1:]))
        # finding the min(left) and max(right) coordinates
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        coordinates = (torch.stack([x1, y1, x2, y2], dim=1)).to(torch.int)
        min_x = torch.min(coordinates[:, 0])
        min_y = torch.min(coordinates[:, 1])
        max_x = torch.max(coordinates[:, 2])
        max_y = torch.max(coordinates[:, 3])
        # return bbox
        return jsonify(result=[min_x, min_y, max_x, max_y])
    
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)