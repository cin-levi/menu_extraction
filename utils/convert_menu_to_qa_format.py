"""
There is slightly different here.
There are some document that is not completed annotating, then we just get the training data from starting page to the
page which contain annotation. Ignore all the rest
Each page can be considered as one document =)))))
Also need to remove the overlap between name and vintage
"""

from processors import prj_path
import os
from glob import glob
import json
from tqdm import tqdm
from uuid import uuid4
from unidecode import unidecode
from utils.qa_utils import is_whitespace

"Convert sroie dataset to Levi's format"


def normalize_boxes(boxes):
    full_text = ''
    bboxes = []
    labels = {}
    for box in boxes:
        box['bbox'] = [float(x) for x in box['bbox']]
        text = unidecode(box['text'].strip())
        # TODO: fix the error here (… -> ...)
        mapping_error = False
        mapping_dict = {}
        if len(text) != len(box['text'].strip()):
            mapping_error = True
            text = ''
            for i, char in enumerate(box['text'].strip()):
                normalized_char = unidecode(char)
                mapping_dict[i] = len(text)
                text += normalized_char
        # Remap the answer due to the conflict
        if mapping_error:
            for annotation in box['annots']:
                if annotation is None:
                    continue
                for key in annotation:
                    for answer in annotation[key]:
                        answer['answer_start'] = mapping_dict[answer['answer_start']]
                        answer['text'] = unidecode(answer['text'])
                        assert text[answer['answer_start']:].startswith(answer['text'])

        # Lower all key
        box['annots'] = [{k.lower(): x[k] for k in x} for x in box['annots'] if x is not None]

        # TODO: remove overlap between wine name and vintage
        other_annots = []
        for annotation in box['annots']:
            if annotation is None:
                continue
            if 'w' in annotation:
                for other_label in annotation:
                    if other_label != 'w':
                        for other_annotation in annotation[other_label]:
                            for wine_annotation in annotation['w']:
                                if other_annotation['text'] in wine_annotation['text']:
                                    # if not wine_annotation['test'].strip().endswith(other_annotation['text'].strip()):
                                    #     another_wine
                                    start = wine_annotation['text'].find(other_annotation['text'])
                                    end = start + len(other_annotation['text'])
                                    wine_annotation['text'] = wine_annotation['text'][:start]
                                    # Check if other annotation is in the middle of the wine
                                    posterious = wine_annotation['text'][end:].strip()
                                    if len(posterious) > 0:
                                        posterious_start = wine_annotation['text'].find(posterious)
                                        assert posterious_start > start
                                        annotation['w'].append({'text': posterious, 'answer_start': posterious_start})

        for annotation in box['annots']:
            if annotation:
                for label in annotation:
                    for answer in annotation[label]:
                        if len(answer['text']) == 0:
                            continue
                        if label in ['wine', 'ww']:
                            label = 'w'
                        elif label == 'vintage':
                            label = 'v'
                        elif label == 'price':
                            label = 'p'
                        if label.strip() not in ['w', 'v', 'p']:
                            if label in ['d', 'e', 'q', '6']:
                                label = 'w'
                            elif label in ['pp', 'pw']:
                                label = 'p'
                            else:
                                print('Label: ', label)
                                print(answer['text'])
                                1 / 0
                        start_sentence = len(full_text) + 1 if len(full_text) > 0 else 0
                        start_answer = start_sentence + answer['answer_start']
                        end = start_answer + len(answer['text'])
                        label = label.strip()
                        if label not in labels:
                            labels[label] = []
                        labels[label].append({'start': start_answer,
                                              'end': end,
                                              'text': answer['text']})
        if len(full_text) > 0:
            full_text += ' '
        full_text += text

        # TODO: split words here
        _word = ''
        _start = None
        word_starts = []
        word_ends = []
        for i, char in enumerate(text):
            if is_whitespace(char):
                if len(_word) > 0:
                    word_starts.append(_start)
                    word_ends.append(i)
                _word = ''
                _start = None
            else:
                if len(_word) == 0:
                    _start = i
                _word += char
        if len(_word) > 0:
            word_starts.append(_start)
            word_ends.append(len(text))


        for i in range(len(word_starts)):
            start = word_starts[i]
            end = word_ends[i]
            x0 = box['bbox'][0] + int((box['bbox'][2] - box['bbox'][0]) / len(text) * start)
            x1 = box['bbox'][0] + int((box['bbox'][2] - box['bbox'][0]) / len(text) * end)
            bboxes.append([x0, box['bbox'][1], x1, box['bbox'][3]])

    min_x = min([x[0] for x in bboxes])
    min_y = min([x[1] for x in bboxes])
    max_x = max([x[2] for x in bboxes])
    max_y = max([x[3] for x in bboxes])
    for box in bboxes:
        box[0] = int((box[0] - min_x) / (max_x - min_x) * 1000)
        box[1] = int((box[1] - min_y) / (max_y - min_y) * 1000)
        box[2] = int((box[2] - min_x) / (max_x - min_x) * 1000)
        box[3] = int((box[3] - min_y) / (max_y - min_y) * 1000)
    return full_text.lower(), bboxes, labels


def read_data(files):
    full_data = {'data': []}
    all_keys = []
    for file in tqdm(files):
        print("file: ", file)
        name = os.path.basename(file)
        boxes = json.load(open(file, 'r', encoding='utf-8'))
        #  Filter page without data here
        pages = sorted(list(set(x['page_number'] for x in boxes)))
        box_in_pages = {p: [] for p in pages}
        for box in boxes:
            box_in_pages[box['page_number']].append(box['annots'][0] is not None)
        page_with_annotation = [p for p in box_in_pages if any(box_in_pages[p])]

        # if len(pages) > 5:
        #     max_page = max(page_with_annotation)
        #     print('max_page: ', max_page)
        #     accepted_pages = [p for p in pages if p <= max_page]
        # else:
        #     accepted_pages = pages
        accepted_pages = page_with_annotation

        for p in accepted_pages:

            _boxes = [box for box in boxes if box['page_number'] == p]
            if len(_boxes) == 0:
                continue
            final_text, bboxes, labels = normalize_boxes(_boxes)
            final_text = unidecode(final_text).lower()
            data = {'title': name, 'paragraphs': [{'context': final_text, 'qas': [], "position": bboxes}]}
            all_keys.extend(labels)
            for label in labels:
                qa = {'question': label,
                      'answers': [],
                      'id': str(uuid4()),
                      'is_impossible': True
                      }
                for answer in labels[label]:
                    if final_text[answer['start']: answer['end']] != unidecode(answer['text']).lower():
                        print("ERROR: ",
                              f"{final_text[answer['start']: answer['end']]} || {unidecode(answer['text']).lower()}")
                        continue
                    qa['answers'].append({'answer_start': answer['start'],
                                          'text': final_text[answer['start']: answer['end']]})
                    qa['is_impossible'] = False
                # else:
                data['paragraphs'][0]['qas'].append(qa)
            full_data['data'].append(data)

    # with open(prj_path + '/data/question_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(sorted(list(set(all_keys))), f, ensure_ascii=False)

    with open(prj_path + '/data/full.json', 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False)


if __name__ == '__main__':
    # files = glob(prj_path + '/data/json/train/*.json')
    files = glob('D:\\menu_extraction\\data\\json\\full/*.json')
    # files = ['D:\\menu_extraction\\data\\json\\train\\397. Lafayette Grand Cafe and Bakery.pdf.json']
    read_data(files)
    pass
