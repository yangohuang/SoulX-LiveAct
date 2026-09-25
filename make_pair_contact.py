"""Render labeled, matched-time contact sheets for separately sampled videos."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PATH"))
    parser.add_argument("--seconds", nargs="+", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    captures = []
    try:
        for label, path in args.video:
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise ValueError(f"cannot open {path}")
            captures.append((label, capture))
        cell_w, cell_h, label_h = 208, 360, 25
        sheet = Image.new("RGB", (cell_w * len(args.seconds),
                                  (cell_h + label_h) * len(captures)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, (label, capture) in enumerate(captures):
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for column, second in enumerate(args.seconds):
                frame_index = round(second * fps)
                if frame_index < 0 or frame_index >= count:
                    raise ValueError(f"{label}: time {second}s outside decoded video")
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                okay, bgr = capture.read()
                if not okay:
                    raise ValueError(f"{label}: cannot decode frame {frame_index}")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                image.thumbnail((cell_w, cell_h))
                x, y = column * cell_w, row * (cell_h + label_h)
                draw.text((x + 3, y + 4),
                          f"{label} / {second:g}s / f{frame_index}", fill="black")
                sheet.paste(image, (x, y + label_h))
    finally:
        for _, capture in captures:
            capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
