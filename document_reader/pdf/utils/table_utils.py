from pdfplumber.page import Page
from pdfplumber.table import TableFinder

from ..table import Table


def extract_tables(page: Page):
    page = page.filter(keep_visible_lines)

    table_finder = TableFinder(page)
    tables = table_finder.tables

    return [Table(table) for table in tables if
            len(table.cells) > 1], table_finder


def keep_visible_lines(obj):
    if obj['object_type'] == 'rect':
        return obj['non_stroking_color'] == 0
    return True
