"""Create a sparse, labeled frame contact sheet for long-rollout visual review."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seconds", nargs="+", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for second in args.seconds:
            index = round(second * fps)
            if index < 0 or index >= count:
                raise ValueError(f"time {second} s lies outside decoded video")
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            okay, bgr = capture.read()
            if not okay:
                raise ValueError(f"cannot decode frame {index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((208, 360))
            frames.append((second, index, image))
    finally:
        capture.release()
    width = max(image.width for _, _, image in frames)
    height = max(image.height for _, _, image in frames)
    sheet = Image.new("RGB", (width * len(frames), height + 32), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (second, index, image) in enumerate(frames):
        x = column * width
        draw.text((x + 4, 6), f"{second:g} s / frame {index}", fill="black")
        sheet.paste(image, (x, 32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
