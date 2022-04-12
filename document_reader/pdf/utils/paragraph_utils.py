import logging
from operator import truth
from typing import List

from pdfplumber.page import Page
from pdfplumber.utils import obj_to_bbox

from .utils import is_contain, extract_lines
from ..line import Line
from ..paragraph import Paragraph
from ..table import Table
from ...utils import normalize_text

logger = logging.getLogger(__name__)


def construct_paragraphs(lines: List[Line], **kwargs):
    from ...utils.config import settings
    paragraphs = []
    paragraph = Paragraph()
    for index, line in enumerate(lines):
        for rule in settings.pdf.constructing.rules:
            try:
                is_included = rule.run(paragraph, lines, index, **kwargs)
                if is_included is not None:
                    if not is_included:
                        paragraphs.append(paragraph)
                        paragraph = Paragraph()
                    paragraph.append(line)

                    visualize(index, lines, rule)
                    break
            except BaseException:
                logger.warning(f'Exception occurred while applying rule '
                               f'"{rule.name}" for line "{line}"')

                if settings.debug.enable:
                    raise
        else:
            logger.warning(f'[Line skipped] There isn\'t any capable rule'
                           f' to process: "{lines[index]}"')

    # Check for the last paragraph
    if paragraph:
        paragraphs.append(paragraph)

    return paragraphs


def extract_paragraphs(page: Page, tables: List[Table]):
    # Remove all objects lie within table
    filtered_page = page.filter(
        lambda obj: not any(is_contain(
            container=table.plumber_table.bbox,
            containee=obj_to_bbox(obj)
        ) for table in tables)
    )
    chars = line_preprocess(filtered_page)
    annots = page.annots

    lines = [Line(page, chars) for chars in extract_lines(chars)]
    from .annotation_utils import add_annotation_to_line, ANNOTATORS
    lines = [add_annotation_to_line(line, annots, ANNOTATORS) for line in lines]
    return [Paragraph(line) for line in lines]


def visualize(index, lines, rule):
    from ...utils.config import settings
    if settings.debug.show_para:
        needed_indent = rule.name.endswith("[+]") and \
                        not rule.name.startswith("FirstLine")
        indentation = lines[index].rect['x0']
        logger.debug(
            f'{"   " if needed_indent else ""}'
            f'"{normalize_text(lines[index].text, remove_space=False)}"'
            f' <- "{rule.name}"'
            f' "I:{indentation}"'
            f' "N:{lines[index].numbering.text}"')


def line_preprocess(page):
    from ...utils.config import settings
    p = settings.pdf.line_preprocess
    return p(page) if p else page.chars


def line_postprocess(lines):
    # Remove empty lines
    lines = list(filter(truth, lines))

    from ...utils.config import settings
    p = settings.pdf.line_postprocess
    return p(lines) if p else lines
