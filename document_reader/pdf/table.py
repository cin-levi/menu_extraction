from pdfplumber.table import Table as PlumberTable
from pdfplumber.utils import bbox_to_rect, within_bbox

from .paragraph import Paragraph as PDFParagraph
from ..paragraph import Paragraph as BaseParagraph


class TableRow(BaseParagraph):

    def __init__(self, table, cells, index=None):
        self.row_index = index
        self.table = table
        self.cells = cells

    @property
    def text(self) -> str:
        from ..utils.config import settings
        deliminator = settings.exporting.cell_deliminator
        return deliminator.join(
            ['' if cell is None else cell for cell in self.cells])

    @property
    def page_number(self) -> int:
        return self.table.page_number

    @property
    def page(self):
        return self.table.plumber_table.page

    @property
    def header_level(self):
        return 0

    @property
    def rect(self):
        return bbox_to_rect(self.table.plumber_table.rows[self.row_index].bbox)

    def is_table(self) -> bool:
        return True

    def expandable(self):
        diff = len(self.cells[-1] or []) - sum(
            len(c or []) for c in self.cells[:-1])
        return diff > 700

    def expand(self):
        from document_reader.pdf.line import Line
        from document_reader.pdf.utils.utils import extract_lines
        from .utils.paragraph_utils import construct_paragraphs

        rows = []
        cells = self.table.plumber_table.rows[self.row_index].cells
        page = self.table.plumber_table.page
        for index, box in enumerate(cells):
            chars = within_bbox(page.chars, box)
            lines = [Line(self.table.plumber_table.page, chars) for chars in
                     extract_lines(chars)]
            if index == len(cells) - 1:
                rows.extend([ExpandedTableRow(index, p.lines) for p in
                             construct_paragraphs(lines, page=page)])
            else:
                rows.append(ExpandedTableRow(index, lines))
        return rows


class ExpandedTableRow(PDFParagraph):
    def __init__(self, column_index, lines=()):
        super().__init__()
        self.lines = lines
        self.column_index = column_index

    def is_table(self) -> bool:
        return True


class Table:
    def __init__(self, plumber_table: PlumberTable):
        self.plumber_table = plumber_table

    def __iter__(self):
        yield from self.rows

    @property
    def rect(self):
        return bbox_to_rect(self.plumber_table.bbox)

    @property
    def page_number(self):
        return self.plumber_table.page.page_number

    @property
    def rows(self):
        extracted_table = self.plumber_table.extract()
        return [TableRow(self, cells, index) for index, cells in
                enumerate(extracted_table)]

    def expanded_rows(self):
        rows = []
        for row in self.rows:
            if row.expandable():
                rows.extend(row.expand())
            else:
                rows.append(row)
        return rows
