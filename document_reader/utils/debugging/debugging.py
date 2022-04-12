import logging
from operator import attrgetter
from pathlib import Path
from typing import List

from pdfplumber.utils import objects_to_rect, obj_to_bbox, bbox_to_rect

logger = logging.getLogger(__name__)


def generate_debug_image(page, paragraphs: List = None, filename: str = None,
                         table_finder=None):
    if paragraphs is None:
        paragraphs = page.chars

    im = page.to_image()

    if table_finder:
        im.debug_tablefinder(table_finder)

    if len(paragraphs) > 0:
        if isinstance(paragraphs[0], dict):
            rects = [bbox_to_rect(obj_to_bbox(p)) for p in paragraphs]
            im.draw_rects(paragraphs)
        else:
            im.draw_rects(map(attrgetter('rect'), paragraphs))

    from ..config import settings
    path = Path(settings.debug.location).joinpath(filename)

    if not Path(path).parent.exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f'Create debugging image: {path}')
    im.save(path)

    return path


def draw_debugging_rects(page, rects, out_path='.debug/debugging_image.png'):
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    im = page.to_image()
    im.draw_rects(rects)
    im.save(out_path)
    from document_reader.utils.utils import launch
    launch(out_path)
