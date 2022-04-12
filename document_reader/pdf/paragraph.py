import itertools
import logging
from typing import List

from pdfplumber.utils import objects_to_rect, objects_to_bbox

from .bullet import Bullet
from .line import Line
from .numbering import Numbering
from .style import ALIGNMENT
from ..paragraph import Paragraph as BaseParagraph

logger = logging.getLogger(__name__)


class Paragraph(BaseParagraph):

    def __init__(self, line=None):
        self.lines: List[Line] = []
        if line is not None:
            self.lines.append(line)

    @property
    def annotation(self):
        return [line.annotation for line in self.lines]

    @property
    def bbox(self) -> List:
        return [f'{x:.2f}' for x in objects_to_bbox(self.chars)]

    @property
    def chars(self):
        return list(itertools.chain(*(line.chars for line in self.lines)))

    @property
    def bullet(self):
        return Bullet(self.text, self.rect['x0'])

    @property
    def numbering(self):
        return Numbering(self.text)

    @property
    def text(self):
        return ''.join(line.text for line in self.lines)

    @property
    def page_number(self) -> int:
        return self.page.page_number

    @property
    def page(self):
        return self.lines[0].page

    @property
    def rect(self):
        return objects_to_rect([line.rect for line in self.lines])

    @property
    def indentation(self) -> float:
        return float(self.rect['x0'])

    @property
    def alignment(self):
        if not self:
            return ALIGNMENT.UNKNOWN

        shortest_line = self.lines[0]
        for line in self.lines:
            if shortest_line.width < line.width:
                shortest_line = line

        return shortest_line.alignment

    def append(self, line_segment):
        self.lines.append(line_segment)
