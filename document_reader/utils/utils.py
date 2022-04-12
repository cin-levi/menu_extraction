import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def launch(path: str or Path):
    """ Launch (open) the file with its associated application.
    :param path: path to the file to be opened.
    """
    import os
    path = Path(path)

    if path.exists():
        cmd = 'open' if os.name == 'posix' else 'xdg-open'
        os.system(f'{cmd} "{os.path.abspath(path)}"')
    else:
        logger.error(f'No such file or directory {path.absolute()}')


def convert_file(from_: str or Path, to_: str or Path):
    from_, to_ = Path(from_), Path(to_)
    logger.info(f'Convert {from_.name} => {to_.name}')

    assert from_.exists(), f'Input file does not exist: {from_}'

    to_.parent.mkdir(exist_ok=True)

    in_suffix = from_.suffix.upper().lstrip('.')
    out_suffix = to_.suffix.upper().lstrip('.')

    assert in_suffix != out_suffix.upper(), \
        f'Output file extension must be different with input file extension ({in_suffix}'

    import subprocess
    result = subprocess.run(
        ['soffice', '--headless', '--convert-to', out_suffix.lower(),
         '--outdir', to_.parent, from_],
        capture_output=True, text=True)

    if result.returncode != 0:
        raise ChildProcessError(
            f'Could not convert {from_.name} to {to_.name}\n'
            f'STDOUT: {result.stdout}'
            f'STDERR: {result.stderr}')

    return to_


def find_first_index_of(iterable, cond, default=None):
    try:
        return next(i for i, x in enumerate(iterable) if cond(x))
    except StopIteration:
        return default


@lru_cache(maxsize=10000)
def normalize_text(text, remove_space=True):
    if isinstance(text, str):
        import re
        import unicodedata

        # normalize japanese characters
        text = unicodedata.normalize('NFKC', text)

        # normalize spaces
        text = re.sub(r'[\s+]',
                      '' if remove_space else ' ',
                      text).strip().lower()

    return text


@lru_cache(10000)
def get_header_level(text: str) -> int:
    from .config import settings
    title_patterns = settings.pdf.header.patterns
    for index, patterns in enumerate(reversed(title_patterns), start=1):
        for pattern in patterns:
            if pattern.search(text) is not None:
                return index

    return 0


def is_cjk(text: str):
    ranges = [
        {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},
        # compatibility ideographs
        {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},
        # compatibility ideographs
        {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},
        # compatibility ideographs
        {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},
        # compatibility ideographs
        {'from': ord(u'\u3040'), 'to': ord(u'\u309f')},  # Japanese Hiragana
        {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Katakana
        {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},
        # cjk radicals supplement
        {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
        {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}
        # included as of Unicode 8.0
    ]

    return any(
        any(r["from"] <= ord(char) <= r["to"] for r in ranges)
        for char in text
    )


def count_kanji(text: str) -> int:
    return sum(len(s) for s in re.findall(r'[\u4e00-\u9fff]+', text))


def show_tree(tree,
              show_title_mark=True,
              show_indentation_value=True,
              show_parser_name=True,
              show_page_number=True,
              show_numbering=True):
    from .config import settings
    if not settings.debug.show_tree:
        return

    logger.info(f'Visualizing document tree...')

    def print_node(node, indent_text: str):
        parent = node.parent
        branching_pipe = ' └──' if node == parent.children[-1] else ' ├──'
        msg = ''
        msg += f'{node.paragraph.page_number:<2} ' if show_page_number else ''
        msg += f'{node.paragraph.indentation:>6.1f}' if show_indentation_value else ''
        msg += f' {"T" if node.paragraph.is_header() else " "}' if show_title_mark else ''

        text = node.paragraph.normalized_text.replace('\n', '')
        msg += f'{indent_text}{branching_pipe}{text}'

        msg += f' << "{node.rule_name}"' if show_parser_name else ''
        msg += f' {node.paragraph.numbering.text}' if show_numbering else ''
        msg += f' [{node.paragraph.header_level}]' if show_numbering else ''

        print(msg)

    def travel_tree(node, exporter, indentation='', parent=None):
        if not node.is_root():
            exporter(node, indentation)

        text = indentation + (
            ' │ ' if parent and node != parent.children[-1] else '   ')

        for child in node.children:
            travel_tree(child, exporter, text, node)

    travel_tree(tree, print_node)

