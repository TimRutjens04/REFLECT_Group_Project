from interfaces.IFrameInput import RgbdFrame
from interfaces.IDetection import DetectionResult
from interfaces.ITracking import TrackingResult


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


def tracking_result_to_pil(frame: RgbdFrame, tracking_result: TrackingResult):
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

    for obj in tracking_result.tracked_objects:
        box = obj.bbox_2d
        label = f"{obj.label} (id={obj.track_id})"
        draw.rectangle(box.tolist(), outline="lime", width=line_width)
        draw.text((box[0], box[1] - font_size - 2), label, fill="lime", font=font)

    return rgb_image


def detection_and_tracking_to_pil(
    frame: RgbdFrame,
    detection_result: DetectionResult,
    tracking_result: TrackingResult,
):
    """Overlay detections (red) and tracked objects (lime) on the same frame."""
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

    for obj in tracking_result.tracked_objects:
        box = obj.bbox_2d
        label = f"{obj.label} (id={obj.track_id})"
        draw.rectangle(box.tolist(), outline="lime", width=line_width)
        draw.text((box[0], box[1] - font_size - 2), label, fill="lime", font=font)

    return rgb_image
