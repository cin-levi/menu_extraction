import itertools
from operator import itemgetter

from pdfplumber.utils import objects_to_bbox, DEFAULT_X_TOLERANCE, to_list, cluster_objects, DEFAULT_Y_TOLERANCE, \
    decimalize


def extract_words(page,
                  x_tolerance=DEFAULT_X_TOLERANCE,
                  y_tolerance=DEFAULT_Y_TOLERANCE,
                  keep_blank_chars=False
                  ):
    x_tolerance = decimalize(x_tolerance)
    y_tolerance = decimalize(y_tolerance)

    def process_word_chars(chars):
        x0, top, x1, bottom = objects_to_bbox(chars)
        return {
            "x0": x0,
            "x1": x1,
            "top": top,
            "bottom": bottom,
            "text": "".join(map(itemgetter("text"), chars)),
            "chars": chars
        }

    def get_line_words(chars, tolerance=DEFAULT_X_TOLERANCE):
        get_text = itemgetter("text")
        chars_sorted = sorted(chars, key=itemgetter("x0"))
        words = []
        current_word = []

        for char in chars_sorted:
            if not keep_blank_chars and get_text(char).isspace():
                if len(current_word) > 0:
                    words.append(current_word)
                    current_word = []
                else:
                    pass
            elif len(current_word) == 0:
                current_word.append(char)
            else:
                last_char = current_word[-1]
                if char["x0"] > (last_char["x1"] + tolerance):
                    words.append(current_word)
                    current_word = []
                current_word.append(char)

        if len(current_word) > 0:
            words.append(current_word)
        processed_words = list(map(process_word_chars, words))
        return processed_words

    chars = to_list(page.chars)
    annots = to_list(page.annots)
    from .annotation_utils import add_annotation_to_char, ANNOTATORS
    chars = add_annotation_to_char(chars, annots, ANNOTATORS)
    doctop_clusters = cluster_objects(chars, "doctop", y_tolerance)

    nested = [get_line_words(line_chars, tolerance=x_tolerance)
              for line_chars in doctop_clusters]

    words = list(itertools.chain(*nested))
    return words
