"""Run with python -m anpr IMAGE --output outputs/run1."""
import argparse
import json
from pathlib import Path
from .pipeline import read_image, recognize


def main():
    parser = argparse.ArgumentParser(description='Contour-based ANPR with English EasyOCR')
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/run'))
    parser.add_argument('--gpu', action='store_true', help='Enable EasyOCR GPU support')
    args = parser.parse_args()
    try:
        image = read_image(args.image)
        import easyocr
        import cv2
        reader = easyocr.Reader(['en'], gpu=args.gpu)
        result, stages = recognize(image, reader)
        args.output.mkdir(parents=True, exist_ok=True)
        for name, stage in stages.items():
            if not cv2.imwrite(str(args.output / f'{name}.png'), stage):
                raise OSError(f'Could not save {name}')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f'ANPR error: {error}\n')


if __name__ == '__main__':
    main()
