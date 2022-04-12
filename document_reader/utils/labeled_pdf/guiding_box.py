from pdfplumber.table import TableFinder
from pdfplumber.utils import obj_to_bbox, objects_to_rect

from .utils import extract_text, populate_with_single_line_boxes
from ...pdf.utils.table_utils import keep_visible_lines
from ...pdf.utils.utils import is_contain


class GuidingBox:
    def __init__(self, annotation=None, page=None):
        """ GuidingBox is a class representing a rectangle box with a guiding
        annotation.
        :param annotation: a single annotation corresponding to the box
        :param page: page cropped with the box
        """
        self.annotation = annotation
        self.page = page.crop(obj_to_bbox(annotation)) \
            if annotation else page
        self.labels = []

    def __repr__(self):
        return f'{self.note} => {self.text}'

    def __str__(self):
        return self.text

    def __contains__(self, bbox):
        return is_contain(
            container=self.bbox(),
            containee=bbox.bbox() if isinstance(bbox, GuidingBox) else bbox,
        )

    def __lt__(self, other):
        this, that = self.doc_rect(), other.doc_rect()
        return this['x0'] < that['x0'] \
            if this['top'] == that['top'] \
            else this['top'] < that['top']

    @property
    def text(self):
        return extract_text(self.page.chars)

    @property
    def page_number(self):
        return self.page.page_number

    @property
    def note(self):
        return (self.annotation['contents'] or 'P').strip().upper()

    @property
    def label_tags(self) -> str:
        return '; '.join(label.tag for label in self.labels)

    @property
    def label_texts(self) -> str:
        return '; '.join(label.text_within_box(self) for label in self.labels)

    def bbox(self):
        return obj_to_bbox(self.annotation)

    def doc_rect(self):
        return objects_to_rect(self.page.chars)


class ParagraphConvertedGuidingBox(GuidingBox):

    def __init__(self, paragraph, note):
        from document_reader.pdf.table import TableRow
        page = paragraph.table.plumber_table.page \
            if isinstance(paragraph, TableRow) else paragraph.lines[0].page
        rect = paragraph.rect
        rect['x1'] = min(rect['x1'], page.width - 1)
        rect['bottom'] = min(rect['bottom'], page.height - 1)
        super().__init__(page=page.crop(obj_to_bbox(rect)))
        self.paragraph = paragraph
        self.note = note

    @property
    def note(self):
        return self.note_

    @note.setter
    def note(self, value):
        self.note_ = value

    def bbox(self):
        return obj_to_bbox(self.doc_rect())


class NestedGuidingBox(GuidingBox):
    def __init__(self, box):
        """ NestedGuidingBox is a GuidingBox containing one or more guiding boxes.
        The box would be annotated with `PC`(a text column in a multi-column page)
        :param box: the original GuidingBox
        """
        super().__init__()
        self.boxes = []
        self.annotation = box.annotation
        self.page = box.page

    @property
    def text(self):
        return '\n'.join(c.text for c in self.boxes)

    def adopt(self, box):
        self.boxes.append(box)

    @property
    def children(self):
        return populate_with_single_line_boxes(self.page, self.boxes)


class SingleLineBox(GuidingBox):

    def __init__(self, chars, page):
        """ SingleLineBox is a GuidingBox, representing a single line of text in the pdf.
        By convention, single line will not be annotated, so no annotation for this box
        :param chars: the list of characters in the text line
        :param page: the page where the text line belong to
        """
        super().__init__(page=page)
        self.chars = chars

    @property
    def note(self):
        return 'P'

    @property
    def text(self):
        from pdfplumber.utils import collate_line
        return collate_line(self.chars)

    def bbox(self):
        from pdfplumber.utils import objects_to_bbox
        return objects_to_bbox(self.chars)

    def doc_rect(self):
        return objects_to_rect(self.chars)


class InterPageGuidingBox(GuidingBox):
    def __init__(self, boxes):
        """ InterPageGuidingBox is a GuidingBox, representing a single paragraph
        placed in more than one consecutive pages.
        :param boxes: list of segments of the paragraph on different pages.
        """
        super().__init__()
        self.boxes = boxes

    @property
    def note(self):
        assert all(b.note.startswith('P') for b in self.boxes)
        return 'P'

    @property
    def text(self):
        return ' '.join(b.text for b in self.boxes)

    @property
    def page_number(self):
        return self.boxes[0].page.page_number


class TableGuidingBox(GuidingBox):
    def __init__(self, annotation=None, page=None):
        """ TableGuidingBox is a GuidingBox, representing a table.
        :param annotation: corresponding annotation for the box
        :param page: page where the table belong to, cropped to the table
        """
        super().__init__(annotation, page)
        self.rows = self.parse_table()

    def parse_table(self):
        page = self.page.filter(keep_visible_lines)
        tables = TableFinder(page).tables
        if tables:
            table = tables[0]
            table_text = table.extract()
            return [
                TableRowGuidingBox(row, table_text[index], self.page, self.note)
                for index, row in enumerate(table.rows)
            ]
        else:
            raise LookupError(f'Could not found anny table for {self.note}')


class TableRowGuidingBox(GuidingBox):
    def __init__(self, row, cell_texts, page, note):
        """ TableRowGuidingBox is an autogenerated GuidingBox base on
        TableGuidingBox. It represents a single row in table.
        """
        super().__init__(page=page)
        self._note = note
        self.row = row
        self.cell_texts = cell_texts
        self.labels = []

    @property
    def note(self):
        return self._note

    def bbox(self):
        return self.row.bbox

    @property
    def text(self):
        return '|'.join(self.cell_texts)
