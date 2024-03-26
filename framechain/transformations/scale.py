from enum import Enum
from typing import Literal
import cv2


class ScalingMode(Enum):
    no_scale = "no_scale"
    scale_to_width = "scale_to_width"
    scale_to_height = "scale_to_height"
    scale_to_longest = "scale_to_longest"
    scale_to_shortest = "scale_to_shortest"
    strict = "strict"
    scale_both = "scale_both"


def scale(img, max_size, min_size, mode: ScalingMode = ScalingMode.strict):
    """Scale an image to fit within a given size range."""

    h, w = img.shape[:2]
    min_h, min_w = min_size
    max_h, max_w = max_size

    h_cond = -1 if h < min_h else 1 if h > max_h else 0
    w_cond = -1 if w < min_w else 1 if w > max_w else 0

    match h_cond, w_cond:
        case -1, -1:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    img = cv2.resize(img, (min_w, int(h * min_w / w)))
                case ScalingMode.scale_to_height:
                    img = cv2.resize(img, (int(w * min_h / h), min_h))
                case ScalingMode.scale_to_longest:
                    img = cv2.copyMakeBorder(
                        img, 0, min_h - h, 0, min_w - w, cv2.BORDER_CONSTANT, value=0
                    )
                case ScalingMode.scale_to_shortest:
                    img = cv2.resize(img, (max_w, max_h))
                    img = img[
                        (max_h - min_h) // 2 : (max_h + min_h) // 2,
                        (max_w - min_w) // 2 : (max_w + min_w) // 2,
                    ]
                case ScalingMode.strict:
                    raise ValueError("Image is too small to be scaled.")
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (min_w, min_h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case -1, 0:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    pass
                case ScalingMode.scale_to_height:
                    img = cv2.resize(img, (w, min_h))
                case ScalingMode.scale_to_longest:
                    img = cv2.copyMakeBorder(
                        img, 0, min_h - h, 0, 0, cv2.BORDER_CONSTANT, value=0
                    )
                case ScalingMode.scale_to_shortest:
                    img = img[(h - min_h) // 2 : (h + min_h) // 2, :]
                case ScalingMode.strict:
                    raise ValueError("Image height is too small to be scaled.")
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (w, min_h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case -1, 1:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    img = cv2.resize(img, (max_w, int(h * max_w / w)))
                case ScalingMode.scale_to_height:
                    img = cv2.resize(img, (int(w * min_h / h), min_h))
                case ScalingMode.scale_to_longest:
                    img = cv2.copyMakeBorder(
                        img, 0, min_h - h, 0, max_w - w, cv2.BORDER_CONSTANT, value=0
                    )
                case ScalingMode.scale_to_shortest:
                    img = img[
                        (h - min_h) // 2 : (h + min_h) // 2,
                        (max_w - w) // 2 : (max_w + w) // 2,
                    ]
                case ScalingMode.strict:
                    raise ValueError("Image height is too small to be scaled.")
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (max_w, min_h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case 0, -1:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    img = cv2.resize(img, (min_w, h))
                case ScalingMode.scale_to_height:
                    pass
                case ScalingMode.scale_to_longest:
                    img = cv2.copyMakeBorder(
                        img, 0, 0, 0, min_w - w, cv2.BORDER_CONSTANT, value=0
                    )
                case ScalingMode.scale_to_shortest:
                    img = img[:, (w - min_w) // 2 : (w + min_w) // 2]
                case ScalingMode.strict:
                    raise ValueError("Image width is too small to be scaled.")
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (min_w, h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case 0, 0:
            match mode:
                case ScalingMode.strict:
                    pass
                case ScalingMode.scale_both:
                    pass
                case _:
                    pass
        case 0, 1:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    img = cv2.resize(img, (max_w, h))
                case ScalingMode.scale_to_height:
                    pass
                case ScalingMode.scale_to_longest:
                    pass
                case ScalingMode.scale_to_shortest:
                    img = img[:, (max_w - w) // 2 : (max_w + w) // 2]
                case ScalingMode.strict:
                    raise ValueError("Image width is too large to be scaled.")
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (max_w, h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case 1, -1:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    img = cv2.resize(img, (min_w, int(h * min_w / w)))
                case ScalingMode.scale_to_height:
                    img = cv2.resize(img, (int(w * max_h / h), max_h))
                case ScalingMode.scale_to_longest:
                    img = cv2.copyMakeBorder(
                        img, max_h - h, 0, 0, min_w - w, cv2.BORDER_CONSTANT, value=0
                    )
                case ScalingMode.scale_to_shortest:
                    img = img[
                        (max_h - h) // 2 : (max_h + h) // 2,
                        (w - min_w) // 2 : (w + min_w) // 2,
                    ]
                case ScalingMode.strict:
                    raise ValueError(
                        "Image height is too large and width is too small to be scaled."
                    )
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (min_w, max_h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case 1, 0:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    pass
                case ScalingMode.scale_to_height:
                    img = cv2.resize(img, (w, max_h))
                case ScalingMode.scale_to_longest:
                    img = cv2.copyMakeBorder(
                        img, max_h - h, 0, 0, 0, cv2.BORDER_CONSTANT, value=0
                    )
                case ScalingMode.scale_to_shortest:
                    img = img[(max_h - h) // 2 : (max_h + h) // 2, :]
                case ScalingMode.strict:
                    raise ValueError("Image height is too large to be scaled.")
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (w, max_h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case 1, 1:
            match mode:
                case ScalingMode.no_scale:
                    pass
                case ScalingMode.scale_to_width:
                    img = cv2.resize(img, (max_w, int(h * max_w / w)))
                case ScalingMode.scale_to_height:
                    img = cv2.resize(img, (int(w * max_h / h), max_h))
                case ScalingMode.scale_to_longest:
                    pass
                case ScalingMode.scale_to_shortest:
                    img = img[
                        (max_h - h) // 2 : (max_h + h) // 2,
                        (max_w - w) // 2 : (max_w + w) // 2,
                    ]
                case ScalingMode.strict:
                    raise ValueError("Image is too large to be scaled.")
                case ScalingMode.scale_both:
                    img = cv2.resize(img, (max_w, max_h))
                case _:
                    raise ValueError(f"Invalid scaling mode: {mode}")
        case _:
            raise ValueError(f"Invalid condition: {h_cond}, {w_cond}")

    return img
