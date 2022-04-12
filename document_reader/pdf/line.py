from pdfplumber.page import Page
from pdfplumber.utils import collate_line, objects_to_rect

from .bullet import Bullet
from .numbering import Numbering


class Line:

    def __init__(self, page: Page, chars):
        self.page = page
        self.chars = chars
        self.annotation = None

    def __bool__(self):
        return bool(self.text)

    def __repr__(self):
        return self.text

    @property
    def bullet(self):
        return Bullet(self.text, self.rect['x0'])

    @property
    def numbering(self):
        return Numbering(self.text)

    @property
    def text(self):
        from document_reader import settings
        return collate_line(self.chars, settings.pdf.tolerance_x).strip()

    @property
    def rect(self):
        return objects_to_rect(self.chars)

    @property
    def annot(self):
        print(self.chars)
        1/0
        print(list(set([x['annotation'] for x in self.chars])))
        return None

    @property
    def width(self):
        return self.rect['x1'] - self.rect['x0']

    @property
    def alignment(self):
        from .utils.utils import guess_alignment
        return guess_alignment(
            page_rect=objects_to_rect(self.page.chars),
            text_rect=objects_to_rect(self.chars)
        )
