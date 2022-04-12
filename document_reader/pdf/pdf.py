import logging
from pathlib import Path

import pdfplumber

from .utils.page_utils import *
from .utils.paragraph_utils import extract_paragraphs
from .utils.table_utils import extract_tables
from ..utils.debugging import generate_debug_image

logger = logging.getLogger(__name__)


def load_pdf(file_path: Path):
    with pdfplumber.open(file_path) as pdf:
        return load_pdf_file_object(file_path, pdf)


def load_pdf_file_object(file_path, pdf):
    pdf_paragraphs = []
    images = []
    for page in pre_process_pdf(pdf.pages):
        logger.info(f'Constructing page {page.page_number}')
        image = page.to_image().original
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        images.append(image)

        # tables, table_finder = extract_tables(page)
        tables = []
        paragraphs = extract_paragraphs(page, tables)


        # from ..utils.config import settings
        # if settings.debug.pdf.gen_image:
        #     image_name = f'{file_path.name}-page-{page.page_number}.png'
        #     generate_debug_image(page, paragraphs, image_name, table_finder)

        # paragraphs = combine(tables, paragraphs)
        # post_process_pdf(paragraphs, pdf_paragraphs)
        pdf_paragraphs.extend(paragraphs)

        if not paragraphs:
            logger.warning(f'Empty page: {page.page_number}')
    remove_empty_paragraph(pdf_paragraphs)
    run_indexing(pdf_paragraphs)
    return pdf_paragraphs, images
