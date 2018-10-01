from contextlib import contextmanager

import cv2

__all__ = [
    'video_capture',
]


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()
