from collections import namedtuple

ALIGNMENT = namedtuple('Alignment',
                       ['LEFT', 'CENTER', 'RIGHT', 'UNKNOWN', 'THRESHOLD']
                       )('LEFT', 'CENTER', 'RIGHT', 'UNKNOWN', 1)


class Font:
    SIZE_DIFF_THRESHOLD = 1

    def __init__(self, name=None, size=None):
        self.name: str = name
        self.size: float = size

    def __eq__(self, other):
        return self.face == other.face and abs(
            self.size - other.size) < self.SIZE_DIFF_THRESHOLD

    @property
    def height(self):
        return self.size

    @property
    def face(self):
        return self.name.split('+')[1] if '+' in self.name else self.name

    def is_bold(self):
        return 'bold' in self.name.lower()

    def is_italic(self):
        return 'italic' in self.name.lower()

    @staticmethod
    def from_char(char: dict):
        name = char['fontname']
        size = char['size']
        return Font(name, size)
