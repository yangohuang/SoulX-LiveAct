"""Show matched within-period mouth frames from a 25-FPS face crop."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-video", type=Path, required=True)
    parser.add_argument("--offset-seconds", type=float, required=True)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    capture = cv2.VideoCapture(str(args.crop_video))
    if not capture.isOpened():
        raise ValueError(f"cannot open {args.crop_video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if abs(fps - 25) > 1e-3:
        raise ValueError("expected a 25-FPS SyncNet crop")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cell_w, cell_h, label_h = 200, 130, 24
    sheet = Image.new("RGB", (cell_w * args.frames,
                              (cell_h + label_h) * args.periods), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        for period in range(args.periods):
            start = round((period * 30 + args.offset_seconds) * fps)
            if start < 0 or start + args.frames > count:
                raise ValueError(f"incomplete mouth review window {start}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for column in range(args.frames):
                okay, bgr = capture.read()
                if not okay:
                    raise ValueError(f"cannot decode crop frame {start + column}")
                rgb = cv2.cvtColor(bgr[88:153, 60:160], cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb).resize((cell_w, cell_h))
                x, y = column * cell_w, period * (cell_h + label_h)
                draw.text((x + 3, y + 4),
                          f"period {period + 1}, frame {start + column}",
                          fill="black")
                sheet.paste(image, (x, y + label_h))
    finally:
        capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
