import json
import logging
import re
from itertools import chain
from pathlib import Path

import pdfplumber
from pdfplumber.utils import to_list, cluster_objects, \
    collate_line, obj_to_bbox, get_bbox_overlap, calculate_area
from pydantic import validate_arguments
from decimal import Decimal

def decimalize(v, q=None):
    # If already a decimal, just return itself
    if type(v) == Decimal:
        return v

    # If tuple/list passed, bulk-convert
    if isinstance(v, (tuple, list)):
        return type(v)(decimalize(x, q) for x in v)
    else:
        return _decimalize(v, q)


logger = logging.getLogger(__name__)


def extract_boxes_from_page(page):
    logger.debug(f'>> Collect label from page {page.page_number}')
    boxes = get_box_from_page(page)
    boxes = gather_to_nested_boxes(boxes)
    boxes = populate_with_single_line_boxes(page, boxes)
    boxes = flatten_nested_boxes(boxes)
    boxes = populate_box_with_tag(boxes, get_tag_from_page(page))
    return boxes


def get_box_from_page(page):
    from .guiding_box import GuidingBox, TableGuidingBox

    boxes = []
    for annotation in page.annots:
        if annotation['data']['Subtype'].name == 'Square':
            note = annotation['contents'] or ''
            cls = TableGuidingBox if note.startswith('T') else GuidingBox
            boxes.append(cls(annotation, page))

    return boxes


def split_note(text: str):
    quote_re = re.compile('"([^"]*)"')
    tags = quote_re.findall(text)
    text = quote_re.sub('', text)
    tags.extend([t for t in text.split(',') if t.strip()])
    return tags


def get_tag_from_page(page):
    from .label import Label

    tags = []
    for annotation in page.annots:
        if annotation['data']['Subtype'].name == 'Highlight':
            note = annotation['contents']
            if not note:
                # from ..debugging.debugging import draw_debugging_rects
                # bbox = obj_to_bbox(annotation)
                # draw_debugging_rects(page, [bbox_to_rect(bbox)], f'.debug/{}')
                logger.warning(
                    f'L05: Cannot parse annotation: "missing content '
                    f'for a highlight on page {page.page_number}"')
            else:
                tags.extend(
                    Label(tag, annotation, page) for tag in split_note(note))
    return tags


def get_tag_from_pdf(pdf):
    tags = []
    for page in pdf.pages:
        tags.extend(get_tag_from_page(page))

    if not tags:
        logger.warning('L03: Missing annotation')

    return tags


def gather_to_nested_boxes(boxes):
    from .guiding_box import NestedGuidingBox
    outer_boxes = []
    nested_boxes = [NestedGuidingBox(b) for b in boxes if b.note == 'PC']
    for box in filter(lambda b: b.note != 'PC', boxes):
        for nested_box in nested_boxes:
            if box in nested_box:
                nested_box.adopt(box)
                break
        else:
            outer_boxes.append(box)

    return sorted(outer_boxes + nested_boxes)


def populate_with_single_line_boxes(page, boxes):
    from .guiding_box import SingleLineBox
    from document_reader.pdf.utils.utils import extract_lines

    page = page.filter(
        lambda x: all(obj_to_bbox(x) not in b for b in boxes)
    )

    lines = [SingleLineBox(chars, page) for chars in extract_lines(page.chars)]
    return sorted(list(boxes) + lines)


def flatten_nested_boxes(boxes):
    from .guiding_box import NestedGuidingBox, TableGuidingBox
    flattened_boxes = []
    for box in boxes:
        if isinstance(box, NestedGuidingBox):
            flattened_boxes.extend(box.children)
        elif isinstance(box, TableGuidingBox):
            flattened_boxes.extend(box.rows)
        else:
            flattened_boxes.append(box)

    return flattened_boxes


def populate_box_with_tag(boxes, labels):
    logger.debug(f'Box: {len(boxes)}')
    logger.debug(f'Label: {len(labels)}')

    for box in boxes:
        for label in labels:
            if label.page_number == box.page_number \
                    and is_bbox_intersected(box.bbox(), label.bbox()):
                box.labels.append(label)
    return boxes


def join_boxes(boxes_by_pages):
    from .guiding_box import InterPageGuidingBox

    boxes = []
    joined_boxes = []
    for box in list(chain(*boxes_by_pages)):
        if box.note in ('PS', 'PM', 'PE'):
            joined_boxes.append(box)
            if box.note == 'PE':
                boxes.append(InterPageGuidingBox(joined_boxes))
        else:
            boxes.append(box)

    return boxes


def is_bbox_intersected(a, b):
    box = get_bbox_overlap(a, b)
    if box is None:
        return False

    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    return width > 1 and height > 2


def extract_text(chars):
    chars = to_list(chars)
    if not chars:
        return ''

    clusters = cluster_objects(chars, "doctop", 1)
    lines = (collate_line(c, 3) for c in clusters)
    return ' '.join(lines).strip()


def build_document_tree(boxes):
    from .node import Node, RootNode
    logger.debug(f'Form document tree base on the label for structure')

    leaf = RootNode()
    for index, box in enumerate(boxes):
        leaf = leaf.add_new_leaf(Node(box, index))

    return leaf.root()


def export_to_xlsx(root, out_file: Path) -> Path:
    from ..excel import Excel
    with Excel(out_file) as excel:
        sheet, formats = excel.add_sheet(
            ['Page Number', 'Text', 'Title', 'Is Title', 'Is Table', 'Index',
             'Parent Index', 'Tag', 'Value'],
            [5, 100, 5, 7, 7, 5, 11, 20, 30]
        )

        for node in root.dfs():
            fmt = formats.title if node.is_header else formats.default
            sheet.write_row(
                (node.box.page_number, fmt),
                (node.box.text, fmt),
                ('', fmt),  # title text
                ('x' if node.is_header else '', fmt),
                ('x' if node.is_table else '', fmt),
                (node.index,
                 formats[node.index] if node.is_header else formats.default),
                (node.parent_index, formats[node.parent_index]),
                node.box.label_tags,
                node.box.label_texts
            )

    return out_file


def export_to_json(root, out_file: Path) -> Path:
    data = [dict(
        page_number=node.box.page_number,
        text=node.box.text,
        title='',
        is_header=node.is_header,
        is_table=node.is_table,
        index=node.index,
        parent_index=node.parent_index,
        tags=node.box.label_tags,
        values=node.box.label_texts
    ) for node in root.dfs()]

    with open(out_file, mode='w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=True)

    return out_file


def extract_boxes(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return join_boxes(extract_boxes_from_page(page) for page in pdf.pages)


def has_label_for_structure(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if get_box_from_page(page):
                return True
    return False


def populate_document_with_tag(root, pdf):
    from document_reader.utils.labeled_pdf.node import RootNode
    from document_reader.pdf.table import TableRow

    def get_note(node):
        if isinstance(node, TableRow):
            return 'T'
        if node.children:
            return 'H'
        return 'P'

    def transform(old, new):
        for child in old.children:
            from document_reader.utils.labeled_pdf.node import Node
            from document_reader.utils.labeled_pdf.guiding_box import \
                ParagraphConvertedGuidingBox
            box = ParagraphConvertedGuidingBox(child.paragraph, get_note(child))
            node = Node(box, child.index)
            transform(child, node)
            new.add_child(node)

    # Copy the tree of paragraph to tree of box
    new_root = RootNode()
    transform(root, new_root)

    # Populate boxes with tag
    tags = get_tag_from_pdf(pdf)
    boxes = [node.box for node in new_root.dfs()]
    populate_box_with_tag(boxes, tags)

    return new_root


def load_labeled_pdf(file_path):
    logger.info(f'Loading labeled pdf from file {file_path.name}')
    if has_label_for_structure(file_path):
        boxes = extract_boxes(file_path)
        return build_document_tree(boxes)
    else:
        from document_reader.pdf.pdf import load_pdf_file_object
        from document_reader.utils import build_document_structure
        with pdfplumber.open(file_path) as pdf:
            paragraphs, _ = load_pdf_file_object(file_path, pdf)
            root = build_document_structure(paragraphs)
            return populate_document_with_tag(root, pdf)


def get_annotated_chars(page, annotation):
    def intersects_bbox(objs, bbox):
        matching = []
        for obj in objs:
            b = get_bbox_overlap(obj_to_bbox(obj), bbox)
            if b and float(calculate_area(b)) > float(
                    calculate_area(obj_to_bbox(obj))) / 1.25:
                matching.append(obj)
        return matching

    chars = page.chars
    quad = decimalize(annotation['data']['QuadPoints'])

    filtered_chars = []
    for index in range(0, 8000, 8):
        q = quad[index:index + 8]
        if not q:
            break
        bbox = (q[4], page.height - q[3], q[2], page.height - q[5])
        filtered_chars.extend(intersects_bbox(chars, bbox))
    return filtered_chars


@validate_arguments
def validate_labeled_pdf(pdf_path: Path):
    """ Error list:
        L01: Corrupted or not a PDF
        L02: The PDF does not have any text
        L03: Missing annotations
        L04: Unsupported language. PDF must be in Japanese or English
        L05: Cannot parse annotation: <detail...>

        L00: Unclassified error: "<detail...>"
    """
    from pdfminer.pdfparser import PDFSyntaxError
    from langid import langid
    from .. import normalize_text
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join(p.extract_text() or '' for p in pdf.pages)
            text = normalize_text(text)

            assert text, "L02: The PDF does not have any text"

            assert langid.classify(text)[0] in ['ja', 'en'], \
                'L04: Unsupported language. PDF must be in Japanese or English'

            has_highlight = False
            for annotation in pdf.annots:
                if annotation['data']['Subtype'].name == 'Highlight':
                    has_highlight = True
                    note = annotation['contents']
                    assert note, \
                        f'L05: Cannot parse annotation: "missing content for ' \
                        f'a highlight on page {annotation["page_number"]}"'

            assert has_highlight, 'L03: Missing annotation'

    except PDFSyntaxError:
        raise AssertionError('L01: Corrupted or not a PDF')
    except BaseException as e:
        if isinstance(e, AssertionError) and str(e).startswith('L0'):
            raise
        else:
            raise AssertionError(f'L00: Unclassified error: "{e}"')
