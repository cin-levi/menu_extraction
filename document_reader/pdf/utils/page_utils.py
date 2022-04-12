from typing import List

from .utils import filter_pages
from ..paragraph import Paragraph
from ..table import Table
from ...utils.utils import find_first_index_of


def combine(tables: List[Table], paragraphs: List[Paragraph]):
    flow = paragraphs.copy()
    tables.sort(key=lambda tbl: tbl.rect['top'])

    mappings = []
    for table in tables:
        index = find_first_index_of(
            paragraphs,
            lambda p: p.rect['top'] > table.rect['top'])
        mappings.append(dict(
            index=index if index is not None else len(paragraphs),
            table=table))

    for mapping in reversed(mappings):
        for row in reversed(mapping['table'].expanded_rows()):
            flow.insert(mapping['index'], row)

    return flow


def run_indexing(pdf_paragraphs):
    # Add index to paragraph
    for index, paragraph in enumerate(pdf_paragraphs):
        paragraph.index = index + 1


def remove_empty_paragraph(paragraphs):
    # Remove empty paragraph
    indices = [
        index
        for index, paragraph in enumerate(paragraphs)
        if not paragraph
    ]

    for index in reversed(indices):
        del paragraphs[index]


def post_process_pdf(page_paragraphs, pdf_paragraphs):
    # TODO: merge inter-pages paragraphs
    pdf_paragraphs.extend(page_paragraphs)


def pre_process_pdf(pages):
    # pages = filter_pages(pages)

    return pages
