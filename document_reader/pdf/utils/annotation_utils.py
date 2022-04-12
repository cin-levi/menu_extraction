from decimal import Decimal
# for visualization
from PIL import Image, ImageDraw

ANNOTATORS = ['rio', 'melanie', 'Rio']
OVERLAP_RATIO = Decimal(0.25)


def check_char_in_box(char, annot):
    x_0, y_0, x_1, y_1 = annot['x0'], annot['y0'], annot['x1'], annot['y1']
    cx_0, cy_0, cx_1, cy_1 = char['x0'], char['y0'], char['x1'], char['y1']
    if not (y_0 <= cy_0 < y_1 or cy_0 <= y_0 < cy_1):
        return False
    if not (x_0 <= cx_0 < x_1 or cx_0 <= x_0 < cx_1):
        return False
    return True


def get_continuous_chars(char_indexes):
    spans = []
    current_span = [char_indexes[0]]
    current_index = char_indexes[0]
    for index in char_indexes[1:]:
        if index == current_index + 1:
            current_span.append(index)
        else:
            if len(current_span) > 0:
                spans.append((current_span[0], current_span[-1]))
            current_span = [index]
        current_index = index
    if len(current_span) > 0:
        spans.append((current_span[0], current_span[-1]))
    return spans


def add_annotation_to_line(line, annots, annotators):
    no_blank_to_correct_char = {len(line.text[:i].replace(' ', '')): i for i in range(len(line.text))}
    for annot in annots:
        if annot['contents']:
            label = annot['contents']
            chars = [c for c in line.chars if len(c['text'].strip()) > 0]
            char_indexes_in_annot = [i for i in range(len(chars)) if check_char_in_box(chars[i], annot)]
            if len(char_indexes_in_annot) > 0:
                spans = get_continuous_chars(char_indexes_in_annot)
                for span in spans:
                    print(line.text)
                    answer_text = line.text[
                                  no_blank_to_correct_char[span[0]]: no_blank_to_correct_char[span[1]] + 1].strip()
                    print(answer_text)
                    answer_start = no_blank_to_correct_char[span[0]]
                    if not line.annotation:
                        line.annotation = {}
                    if label not in line.annotation:
                        line.annotation[label] = []
                    line.annotation[label].append({'text': answer_text, 'answer_start': answer_start})
    return line
