import itertools
import logging
import shutil
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool, cpu_count, Value
from pathlib import Path
from typing import Tuple, List, Dict

from diff_match_patch import diff_match_patch

from .document import Document
from .metric import Metric
from .mismatch import Mismatch
from ..excel import Excel
from ..utils import is_cjk, normalize_text

EMPTY_STR = ' '

DIFF_EDIT_COST = 60
DIFF_TIMEOUT = 60
DIFF_EQUAL = diff_match_patch.DIFF_EQUAL
DIFF_INSERT = diff_match_patch.DIFF_INSERT
DIFF_DELETE = diff_match_patch.DIFF_DELETE

counter = Value('i', 0)
logger = logging.getLogger(__name__)


def get_prefix(segment: Tuple[int, str] = None, max_prefix_length=5):
    if segment is None:
        return EMPTY_STR

    ind, text = segment
    if ind != 0:
        return EMPTY_STR

    prefix = []
    for c in reversed(list(text)):
        if not is_cjk(c):
            break
        prefix.append(c)

        if len(prefix) > max_prefix_length:
            break

    return ''.join(reversed(prefix)) if prefix else EMPTY_STR


def get_suffix(segment: Tuple[int, str] = None, max_suffix_length=5):
    if segment is None:
        return EMPTY_STR

    ind, text = segment
    if ind != 0:
        return EMPTY_STR

    suffix = []
    for c in list(text):
        if not is_cjk(c):
            break

        suffix.append(c)

        if len(suffix) > max_suffix_length:
            break

    return ''.join(suffix) if suffix else EMPTY_STR


def write_dictionary(mismatches: List[Mismatch], text_dr: str, text_ca: str,
                     out_dir: str) -> Path:
    report_file = Path(out_dir) / f'dr_dictionary.xlsx'
    report_file.parent.mkdir(exist_ok=True)

    logger.debug(f'Writing dictionary to {report_file}')
    with Excel(report_file) as excel:
        sheet, fmt = excel.add_sheet(
            ['DR', 'Duplicate', 'CA', 'Duplicate', 'DR in CA', 'Appear in',
             'Duplicate', ],
            [10, 7, 10, 7, 7, 30, 7, ]
        )

        for m in mismatches:
            count = text_ca.count(m.text1)
            sheet.write_row(
                (m.prefix, fmt.red, m.text_del, fmt.default, m.suffix),
                text_dr.count(m.text1),
                (m.prefix, fmt.blue, m.text_ins, fmt.default, m.suffix),
                text_ca.count(m.text2),
                (count, fmt.red if count else fmt.blue),
                str(set(m.ca_name)),
                len(m.ca_name),
            )

    return report_file


def map_with_ca(dr: Path, ca: Path) -> List[Tuple[Path, Path]]:
    dr_files = list(dr.glob('*.xlsx')) if dr.is_dir() else [dr]
    ca_files = list(ca.glob('*.xlsx')) if ca.is_dir() else [ca]

    mapping = []
    for dr_file in dr_files:
        for ca_file in ca_files:
            if dr_file.stem == ca_file.stem:
                mapping.append((dr_file, ca_file))
                break

            if dr_file.stem.split('.')[0] == ca_file.stem.split('.')[0]:
                mapping.append((dr_file, ca_file))
                break

    logger.debug(f'Mapped {len(mapping)}/{len(dr_files)} files')
    return mapping


def squash_mismatches(mismatches: List[Mismatch]) -> List[Mismatch]:
    d = {}
    for mismatch in mismatches:
        if mismatch not in d:
            d[mismatch] = [mismatch.ca_name]
        else:
            d[mismatch].append(mismatch.ca_name)

    for k, v in d.items():
        k.ca_name = v
        yield k


def collect_mismatches(dr: Path, ca: Path) -> Tuple[List[Mismatch], str, str]:
    global counter
    with counter.get_lock():
        counter.value -= 1
        logger.debug(
            f'[{counter.value}] Collect dictionary in {dr.name} <---> {ca.name}')

    text_dr, text_ca = Document(dr).text, Document(ca).text
    diff = get_diff(text_dr, text_ca)
    mismatches = []
    for index in range(len(diff) - 1):
        d1, text1 = diff[index]
        d2, text2 = diff[index + 1]
        if d1 + d2 == 0 and len(text2) + len(text1) <= 5 and is_cjk(
                text2) and is_cjk(text1):
            prefix = get_prefix(diff[index - 1] if index >= 1 else None)
            suffix = get_suffix(
                diff[index + 2] if index + 2 < len(diff) else None)
            mismatch = Mismatch(prefix[-1:], suffix[:1], text1, text2, ca.name)
            mismatches.append(mismatch)

    return mismatches, text_dr, text_ca


def gen_dictionary(dr: str or Path, ca: str or Path, out_dir='.report'):
    """ Generate dictionary to correct OCR mistake.
    Output will be an excel file named `dr_dictionary.xlsx`, placed under a directory specified by `out_dir` parameter.
    The output excel file will have these columns:
      - DR: The wrong OCRed text appeared in lib-document-reader output excels
      - Duplicate: the number of time that the text in column `DR` appear in all of the lib-document-reader output excels
      - CA: The corresponding correct text found in the CA
      - Duplicate: the number of time that the text in `CA` column appear in all of the excel files in CA directory
      - DR in CA: the number of time that the text in `DR` column appear in any of the CA
      - Appear in: the list of exel files that where the text in `CA` column being extracted
      - Duplicate: the number of time that the pair of text in `DR` and `CA` columns appears
    Note: if `DR in CA` is greater than `0` then it is not safe to replace the wrong text(the text in `DR` column) with
    the correct text(the text in `CA` column)
    :param dr: path to a file or directory holding excel file outputted by lib-document-reader
    :param ca: path to a file or directory holding the CA(Corrected Answer)
    :param out_dir: path to a directory to place the to-be-generated dictionary, default: `.report`
    """
    mapping = map_with_ca(Path(dr), Path(ca))
    global counter
    counter = Value('i', len(mapping))
    with Pool(cpu_count()) as p:
        results = p.starmap(collect_mismatches, mapping)

    mismatches = squash_mismatches(
        list(itertools.chain(*[r[0] for r in results])))
    write_dictionary(
        mismatches,
        ''.join(r[1] for r in results),
        ''.join(r[2] for r in results),
        out_dir,
    )


def gen_diff_as_html(excel1: str or Path, excel2: str or Path,
                     out_dir='.report/html'):
    """ Save the diff of the two input excel file to html format.
     The excel must be in standard output format of lib-document-reader
    :param excel1: a path to a file or a directory representing the first excel input
    :param excel2: a path to a file or a directory representing the second excel input
    :param out_dir: a path to a directory where the output html will be placed
    """
    from .document import Document
    mapping = map_with_ca(Path(excel1), Path(excel2))
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    for f1, f2 in mapping:
        html_file = out_dir / f'{Path(f1).stem}.html'

        with open(html_file, mode='w', encoding='utf8') as f:
            d = diff_match_patch()
            d.Diff_Timeout = DIFF_TIMEOUT
            d.Diff_EditCost = DIFF_EDIT_COST

            diff = d.diff_main(Document(f1).text, Document(f2).text,
                               checklines=False)
            d.diff_cleanupSemanticLossless(diff)

            f.write(diff_match_patch().diff_prettyHtml(diff))


def save_excel_as_text(*args, out_dir='.report/text'):
    """ Save excel as text. The excel file must be in the standard output format of lib-document-reader.
    :param args: arguments which hold path to excel file
    :param out_dir: a path to a directory where the to-be-generated text file will be placed, default: `.report/text`
    """
    from .document import Document
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in args:
        excels = [path] if Path(path).is_file() else Path(path).glob('*.xlsx')
        for excel in excels:
            with open(out_dir / f'{Path(excel).name}.txt', mode='w',
                      encoding='utf8') as f:
                f.write(Document(excel).text)


@lru_cache(maxsize=128)
def get_diff(text1, text2):
    d = diff_match_patch()
    d.Diff_Timeout = DIFF_TIMEOUT
    d.Diff_EditCost = DIFF_EDIT_COST

    diff = d.diff_main(text1, text2, checklines=False)
    d.diff_cleanupSemanticLossless(diff)

    return diff


def correct_ocr_error(doc_ca: Document, doc_dr: Document):
    text = ''
    for di, dt in get_diff(doc_dr.text, doc_ca.text):
        if di == DIFF_EQUAL:
            text += dt.replace('\n', '<EOP>')
        elif di == DIFF_DELETE:
            text += '<EOP>' * str(dt).count('\n')
        else:
            text += dt

    ps = text.split('<EOP>')
    for index, paragraph in enumerate(doc_dr):
        paragraph.corrected_text = normalize_text(ps[index])


def write_report(result) -> Dict:
    out_dir = Path(result['out_dir'])
    out_dir.mkdir(exist_ok=True, parents=True)

    filename = result['dr'].file.stem
    for key in ('dr', 'ca'):
        link = out_dir / f'{filename}_{key}.xlsx'
        shutil.copyfile(src=result[key].file, dst=link)

    with Excel(out_dir / f'{filename}_report.xlsx') as excel:
        write_ocr_sheet(excel, result)
        write_paragraph_reconstruction_sheet(excel, result)
        write_document_tree_sheet(excel, result)
        write_metrics(excel, result)
        return result


def write_ocr_sheet(excel: Excel, result):
    sheet, formats = excel.add_sheet(
        ['Page', 'Index', 'Text DR', 'Text CA', 'Wrong char count',
         'Total char count', 'Accuracy'],
        [5, 5, 70, 70, 15, 15, 7],
        'OCR'
    )

    metric = Metric()
    for paragraph in result['dr']:
        dr_text, ca_text = [], []
        char_counter = defaultdict(int)
        for indicator, text in get_diff(paragraph.text,
                                        paragraph.corrected_text):
            char_counter[indicator] += len(text)

            if indicator == DIFF_DELETE:
                dr_text.extend((formats.red, text))
            elif indicator == DIFF_INSERT:
                ca_text.extend((formats.dark_violet, text))
            else:
                dr_text.extend((formats.default, text))
                ca_text.extend((formats.default, text))

        total_char = sum(char_counter.values())
        edited_char = char_counter[DIFF_DELETE] + char_counter[DIFF_INSERT]

        sheet.write_row(paragraph.page_number, paragraph.index, tuple(dr_text),
                        tuple(ca_text), edited_char,
                        round(100 * edited_char / total_char, 2))

        metric.add_negative(edited_char)
        metric.add_positive(total_char - edited_char)

    result['ocr'] = metric


def write_paragraph_reconstruction_sheet(excel: Excel, result):
    sheet, formats = excel.add_sheet(
        ['Page', 'Index', 'Text DR', 'Text CA', 'Accuracy'],
        [5, 5, 100, 100, 7],
        'Paragraph Reconstruction'
    )
    index_ca = 0
    text_length_dr, text_length_ca = 0, 0
    ca_paragraphs = result['ca'].paragraphs

    metric = Metric()
    for paragraph in result['dr']:

        accuracy = 0
        text_ca = ca_paragraphs[index_ca].str() if index_ca < len(
            ca_paragraphs) else ''
        fmt = formats.red
        if paragraph.str() == text_ca:
            fmt = formats.default
            index_ca += 1
            accuracy = 1
        else:
            text_length_dr += len(paragraph.str())

            if text_length_ca >= text_length_dr:
                text_ca = ''
            else:
                text_length_ca += len(text_ca)
                index_ca += 1

        if paragraph.str() + text_ca:
            metric.add(accuracy)

            sheet.write_row(
                paragraph.page_number,
                paragraph.index,
                (paragraph.str(), fmt),
                text_ca,
                accuracy
            )

            while text_length_ca < text_length_dr and index_ca < len(
                    ca_paragraphs):
                text_ca = ca_paragraphs[index_ca].str()
                text_length_ca += len(text_ca)
                sheet.write(text_ca, formats.dark_violet, col_index=3)
                index_ca += 1

    result['para'] = metric


def write_document_tree_sheet(excel, result):
    sheet, formats = excel.add_sheet(
        ['Page', 'Text DR', 'Parent', 'Correct Parent', 'Accuracy'],
        [5, 100, 50, 50, 7],
        'Document Tree'
    )
    dr = result['dr']
    ca = result['ca']

    def get_parent(text):
        for p in ca:
            if text == p.str():
                return str(ca.get(p.parent_index))
        for p in ca:
            if text in p.str():
                return str(ca.get(p.parent_index))

    # Correct document title
    for p in ca:
        if p.parent_index:
            break
        else:
            dr.paragraphs[0].corrected_text = p.text

    metric = Metric()
    for paragraph in dr:
        parent_dr = str(dr.get(paragraph.parent_index))
        parent_ca = get_parent(paragraph.str())

        accuracy = int(parent_dr == parent_ca)

        if paragraph.str():
            metric.add(accuracy)
            sheet.write_row(
                paragraph.page_number,
                (paragraph.str(), formats.default if accuracy else formats.red),
                None if accuracy else (parent_dr,
                                       formats.default if accuracy else formats.dark_violet),
                None if accuracy else parent_ca,
                accuracy
            )

    result['tree'] = metric


def write_metrics(excel, result):
    sheet, _ = excel.add_sheet(
        ['Metric', 'Value'],
        [20, 7],
        'Metrics'
    )

    for key in (
            'ocr',
            'para',
            'tree',):
        sheet.write_row(key, result[key].accuracy)
        # logger.info(
        #     f'{key}: {result[key].accuracy} % - {result["dr"].file.name}')


def write_accumulated_report(reports: List[Dict], out_dir, register_output):
    if not reports:
        return

    from ..managing_experiment import exp
    exp.init()

    with Excel(Path(out_dir) / 'report.xlsx') as excel:
        keys = ('ocr', 'para', 'tree')
        sheet, _ = excel.add_sheet(
            ['Filename', *map(str.upper, keys)],
            [50, 17, 17, 17],
            'By file'
        )

        for report in reports:
            filename = report['dr'].file.name
            sheet.write_row(
                filename,
                *(report[key].accuracy for key in keys))

            for key in keys:
                exp.register(key, filename, report[key].accuracy)

        sheet, _ = excel.add_sheet(
            ['Metric', 'By paragraph', 'By file'],
            [27, 12, 12],
            'Summary'
        )

        for key in keys:
            sheet.write_row(
                key,
                sum((r[key] for r in reports), Metric()).accuracy,
                round(sum((r[key].accuracy for r in reports)) / len(reports), 2)
            )
    exp.set_artifact(out_dir)
    if register_output:
        exp.finish()
