"""Create a synthetic fixture; this is not a real vehicle or benchmark."""
from pathlib import Path
import cv2
import numpy as np

image = np.zeros((240, 480, 3), dtype=np.uint8)
cv2.rectangle(image, (100, 80), (380, 160), (255, 255, 255), -1)
cv2.putText(image, 'DEMO 123', (115, 135), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, (0, 0, 0), 3)
Path('outputs').mkdir(exist_ok=True)
if not cv2.imwrite('outputs/synthetic-demo.png', image):
    raise OSError('Could not save synthetic image')
print('Created outputs/synthetic-demo.png (synthetic fixture, not a benchmark)')
