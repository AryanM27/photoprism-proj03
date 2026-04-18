import io
import random
from uuid import uuid4

from PIL import Image, ImageEnhance


def augment_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Apply random augmentations and return (augmented_bytes, suggested_filename).

    Augmentations (all randomised):
    - Slight rotation: -5 to +5 degrees, expand=False
    - Random horizontal flip (50% chance)
    - Brightness jitter: 0.85-1.15
    - Contrast jitter: 0.85-1.15
    - Random crop: 90-100% of image area, then resize back
    - JPEG recompress at quality 80-95

    Returns bytes of final JPEG and a unique filename like
    "aug_{hex8}.jpg" where hex8 is 8 chars of uuid4.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size

    # Slight rotation
    angle = random.uniform(-5.0, 5.0)
    img = img.rotate(angle, expand=False)

    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Brightness jitter
    brightness_factor = random.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Contrast jitter
    contrast_factor = random.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Random crop (90-100% of image area) then resize back
    crop_ratio = random.uniform(0.90, 1.00)
    w, h = img.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize(original_size, Image.LANCZOS)

    # JPEG recompress
    quality = random.randint(80, 95)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)

    hex8 = uuid4().hex[:8]
    filename = f"aug_{hex8}.jpg"
    return buf.read(), filename
