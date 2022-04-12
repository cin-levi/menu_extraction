from typing import List

from .bullet import Bullet
from .numbering import Numbering


class Paragraph:
    def __bool__(self):
        return bool(self.text)

    def __repr__(self):
        return self.text

    @property
    def bbox(self) -> List:
        return [0, 0, 0, 0]

    @property
    def annotation(self) -> List:
        return []

    @property
    def bullet(self) -> Bullet:
        return Bullet()

    @property
    def numbering(self) -> Numbering:
        return Numbering()

    @property
    def text(self) -> str:
        return ''

    @property
    def normalized_text(self) -> str:
        from .utils import utils
        return utils.normalize_text(self.text, remove_space=False)

    @property
    def indentation(self) -> int or float:
        return 0

    @property
    def title(self) -> str:
        return ''

    @property
    def page_number(self) -> int:
        return 0

    @property
    def header_level(self):
        from .utils.utils import get_header_level
        return get_header_level(self.normalized_text)

    def is_header(self) -> bool:
        return self.header_level > 0

    def is_table(self) -> bool:
        return False
