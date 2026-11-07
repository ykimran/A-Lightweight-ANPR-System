"""Image pipeline adapted from the supplied course project."""
from pathlib import Path
import cv2
import numpy as np


def localize(image):
    """Return OCR crop, crop origin, optional contour, and stage images."""
    if image is None or image.size == 0:
        raise ValueError('A nonempty BGR image is required')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    selected = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0:
            selected = approx
            break
    stages = {'grayscale': gray, 'filtered': filtered, 'edges': edges}
    if selected is None:
        return filtered, (0, 0), None, stages
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [selected], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(selected)
    stages['masked'] = cv2.bitwise_and(image, image, mask=mask)
    crop = filtered[y:y+h, x:x+w]
    stages['crop'] = crop
    return crop, (x, y), selected, stages


def recognize(image, reader):
    """Recognize text and translate crop-relative OCR boxes to image coordinates."""
    crop, (x, y), contour, stages = localize(image)
    detections = []
    annotated = image.copy()
    for box, text, confidence in reader.readtext(crop):
        box = [[float(px) + x, float(py) + y] for px, py in box]
        text = str(text).upper()
        detections.append({'text': text, 'confidence': float(confidence), 'box': box})
        points = np.asarray(box, dtype=np.int32)
        cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
        cv2.putText(annotated, text, (int(box[0][0]), max(15, int(box[0][1])-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if contour is not None:
        cv2.drawContours(annotated, [contour], -1, (255, 180, 0), 2)
    result = {'text': ' '.join(d['text'] for d in detections),
              'fallback': contour is None,
              'plate_contour': contour.reshape(-1, 2).tolist() if contour is not None else None,
              'detections': detections}
    stages['annotated'] = annotated
    return result, stages


def read_image(path):
    image = cv2.imread(str(Path(path)))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image
