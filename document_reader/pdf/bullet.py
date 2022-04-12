import logging

from ..bullet import Bullet as BaseBullet

logger = logging.getLogger(__name__)


class Bullet(BaseBullet):
    def __init__(self, text: str, indentation: int):
        self.char: str = _parse(text)
        self.indentation: int = indentation

    def __eq__(self, other):
        return self.char == other.char and self.indentation == other.indentation

    @property
    def text(self):
        return self.char


def _parse(text: str):
    try:
        from ..utils.config import settings
        text = text.strip()
        if text:
            char = text.strip()[0]
            if char in settings.pdf.bullet.patterns:
                return char
    except BaseException as e:
        logger.exception(f'Failed to parse bullet {e}')
