from pdfplumber.utils import obj_to_bbox, within_bbox

from .utils import extract_text, get_annotated_chars
from ..utils import normalize_text


class Label:
    def __init__(self, tag, annotation, page):
        self.tag = normalize_text(tag, remove_space=False)
        self.annotation = annotation
        self.page = page
        self.chars = get_annotated_chars(page, annotation)

    def __repr__(self):
        return f'{self.tag}: {self.text}'

    @property
    def text(self):
        return extract_text(self.chars)

    @property
    def page_number(self):
        return self.page.page_number

    def text_within_box(self, box):
        return extract_text(within_bbox(self.chars, box.bbox()))

    def bbox(self):
        return obj_to_bbox(self.annotation)
