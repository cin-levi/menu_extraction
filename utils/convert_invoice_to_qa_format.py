from processors import prj_path
import os
from glob import glob
import json
from tqdm import tqdm
from uuid import uuid4
from unidecode import unidecode

"Convert sroie dataset to Levi's format"


def normalize_boxes(boxes):
    full_text = ''
    bboxes = []
    labels = {}
    for box in boxes:
        box['bbox'] = [float(x) for x in box['bbox']]
        text = unidecode(box['text'].strip())
        print("text:   ", text)
        words = text.split()

        text = ' '.join(words)
        for annotation in box['annots']:
            if annotation:
                for label in annotation:
                    if label == 'invoice_company':
                        continue
                    for answer in annotation[label]:
                        if label == 'Totak':
                            label = 'Total'
                        if label == 'ship':
                            label = 'shipping'
                        if label == 'subtotal':
                            label = 'sub_total'
                        if label == 'bill_name' or label == 'bill_company':
                            label = 'bill_person'
                        if label == 'invoice_no':
                            label = 'invoice_number'
                        if label == 'ship_addres':
                            label = 'ship_address'
                        if label.strip() == 'id':
                            label = 'invoice_date'
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

        for i in range(len(words)):
            start = len(' '.join(words[:i]))
            end = len(' '.join(words[:i + 1]))
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


def read_data(folder):
    files = glob(folder + '/*.json')
    full_data = {'data': []}
    all_keys = []
    for file in tqdm(files):
        print("file: ", file)
        name = os.path.basename(file)
        boxes = json.load(open(file, 'r', encoding='utf-8'))
        final_text, bboxes, labels = normalize_boxes(boxes)
        final_text = unidecode(final_text).lower()
        data = {'title': name, 'paragraphs': [{'context': final_text, 'qas': [], "position": bboxes}]}
        if 'id' in labels:
            print(labels)
            1/0
        all_keys.extend(labels)
        for label in labels:
            qa = {'question': label,
                  'answers': [],
                  'id': str(uuid4()),
                  'is_impossible': True
                  }
            for answer in labels[label]:
                if final_text[answer['start']: answer['end']] != unidecode(answer['text']).lower():
                    print("ERROR: ", f"{final_text[answer['start']: answer['end']]} || {answer['text']}")
                    continue
                qa['answers'].append({'answer_start': answer['start'],
                                      'text': final_text[answer['start']: answer['end']]})
                qa['is_impossible'] = False
            # else:
            data['paragraphs'][0]['qas'].append(qa)
        full_data['data'].append(data)

    with open(prj_path + '/data/invoice/question_list.json', 'w', encoding='utf-8') as f:
        json.dump(sorted(list(set(all_keys))), f, ensure_ascii=False)

    with open(prj_path + '/data/invoice/train.json', 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False)


if __name__ == '__main__':
    read_data(prj_path + '/data/invoice/train')
    pass
