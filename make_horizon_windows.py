"""Render selected eight-frame boundary windows for visual error review."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--starts", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"cannot open {args.video}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if any(start < 0 or start + 7 >= count for start in args.starts):
        raise ValueError("a requested eight-frame window is incomplete")
    cell_w, cell_h, label_h = 180, 312, 25
    sheet = Image.new("RGB", (8 * cell_w, len(args.starts) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        for row, start in enumerate(args.starts):
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for column in range(8):
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"cannot decode frame {start + column}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((cell_w, cell_h))
                x, y = column * cell_w, row * (cell_h + label_h)
                draw.text((x + 3, y + 4), f"frame {start + column}", fill="black")
                sheet.paste(image, (x, y + label_h))
    finally:
        capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
