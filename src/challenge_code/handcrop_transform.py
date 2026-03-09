import numpy as np
from PIL import Image
import mediapipe as mp


class HandCropMP:
    """
    Pickle-safe para DataLoader con num_workers > 0:
    - NO crea mp Hands en __init__
    - lo crea de forma lazy dentro del worker la primera vez que se llama
    """
    def __init__(
        self,
        margin: float = 0.25,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        fallback: str = "center_crop",
    ):
        self.margin = margin
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.fallback = fallback

        self._hands = None  # <-- importante: se crea luego

    def _get_hands(self):
        if self._hands is None:
            # Se instancia dentro del proceso worker (ya no necesita pickle)
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=self.max_num_hands,
                model_complexity=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._hands

    def _center_crop(self, rgb: np.ndarray, frac: float = 0.70) -> Image.Image:
        H, W, _ = rgb.shape
        ch, cw = int(H * frac), int(W * frac)
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        return Image.fromarray(rgb[y0:y0 + ch, x0:x0 + cw])

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.array(img.convert("RGB"))
        H, W, _ = rgb.shape

        hands = self._get_hands()
        res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            if self.fallback == "center_crop":
                return self._center_crop(rgb)
            return img

        xs, ys = [], []
        for hand in res.multi_hand_landmarks:
            for lm in hand.landmark:
                xs.append(lm.x)
                ys.append(lm.y)

        x_min = max(0, int((min(xs) - self.margin) * W))
        x_max = min(W, int((max(xs) + self.margin) * W))
        y_min = max(0, int((min(ys) - self.margin) * H))
        y_max = min(H, int((max(ys) + self.margin) * H))

        if x_max <= x_min + 10 or y_max <= y_min + 10:
            if self.fallback == "center_crop":
                return self._center_crop(rgb)
            return img

        crop = rgb[y_min:y_max, x_min:x_max]
        return Image.fromarray(crop)