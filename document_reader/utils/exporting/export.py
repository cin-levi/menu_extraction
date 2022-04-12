from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document_reader import Document

logger = logging.getLogger(__name__)


def export_document_to_json(document: Document, out_file: str) -> str:
    with open(out_file, 'w', encoding='utf8') as f:
        json.dump([
            dict(
                page_number=paragraph.page_number,
                text=paragraph.text,
                bbox=paragraph.bbox,
                annots=paragraph.annotation,
            )
            for paragraph in document.paragraphs
        ], f, ensure_ascii=False, indent=2)

    return out_file


def export_document_to_xlsx(document: Document, out_file: str) -> str:
    from ..excel import Excel
    with Excel(out_file) as excel:
        sheet, formats = excel.add_sheet(
            ['Page Number', 'Text', 'Title', 'Is Title', 'Is Table', 'Index',
             'Parent Index'],
            [5, 100, 30, 15, 15, 15, 15]
        )

        for node in document.tree():
            fmt = formats.title if node.is_header else formats.default
            sheet.write_row(
                (node.page_number, fmt),
                (node.text, fmt),
                (node.title, fmt),
                ('x' if node.is_header else '', fmt),
                ('x' if node.is_table else '', fmt),
                (node.index,
                 formats[node.index] if node.is_header else formats.default),
                (node.parent_index, formats[node.parent_index])
            )

    return out_file


def export(document: Document, extension: str = '.json') -> str:
    """ Export the paragraphs in `document` to `fmt` format in `out_dir` folder.
    :param document: a Document object, typically, output from the document_reader module.
    :param extension: string, supported format: .json, .xlsx
    """
    from ..config import settings
    out_path = Path(settings.exporting.location)
    out_path.mkdir(exist_ok=True, parents=True)

    out_file = str(out_path.joinpath(Path(document.file_path).name + extension))

    logger.info(f'Exporting {document.file_path} to {out_file}')

    try:
        if extension == '.json':
            export_document_to_json(document, out_file)
        elif extension == '.xlsx':
            export_document_to_xlsx(document, out_file)
        else:
            raise Exception(
                f'{extension} format does not support. No file will be exported!')

        return out_file
    except Exception:
        logging.exception(
            f'Context: {__name__}(document={document.file_path},fmt={extension},out_dir={out_path.absolute()})')

        raise
