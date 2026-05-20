from interfaces.IFrameInput import RgbdFrame
from interfaces.IDetection import DetectionResult


def rgb_frame_to_pil(frame: RgbdFrame):
    from PIL import Image

    rgb_image = Image.fromarray(frame.rgb)
    return rgb_image


def detection_result_to_pil(frame: RgbdFrame, detection_result: DetectionResult):
    from PIL import Image, ImageDraw, ImageFont

    rgb_image = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(rgb_image)

    scale = max(rgb_image.size) / 1000
    line_width = max(1, round(scale * 2))
    font_size = max(10, round(scale * 20))

    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    for det in detection_result.detections:
        box = det.bbox_2d
        draw.rectangle(box.tolist(), outline="red", width=line_width)
        draw.text((box[0], box[1] - font_size - 2), det.label, fill="red", font=font)

    return rgb_image
