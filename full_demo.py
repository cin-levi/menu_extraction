import json, tempfile, PyPDF2

from document_reader import Document
from models.inference import LayoutLMInference, LayoutLMV2Inference
from unidecode import unidecode
from uuid import uuid4
from utils.convert_invoice_to_qa_format import normalize_boxes
from models.sort_entities import sort_entities


class LayoutLMDemo(object):
    def __init__(self, model_path, version='v1'):
        if version == 'v1':
            self.model = LayoutLMInference.load_from_checkpoint(model_path)
        else:
            self.model = LayoutLMV2Inference.load_from_checkpoint(model_path)
        self.label_dict = {'w': 'wine_name',
                           'v': 'vintage',
                           'p': 'price'}
        pass

    def write_output(self, clustered_items, other_entities):
        output_boxes = []
        output = []
        for e in other_entities:
            _wine = {}
            if e['entity_type'] in self.label_dict:
                label = self.label_dict[e['entity_type']]
            else:
                label = e['entity_type']
            if label not in _wine:
                _wine[label] = ''
            _wine[label] += ' ' + e['text']
            output_boxes.append(e)

        for cluster in clustered_items:
            a_item = {}
            for e in cluster:
                label = e['entity_type']
                if label not in a_item:
                    a_item[label] = ''
                a_item[label] += ' ' + e['text']
                output_boxes.append(e)
            output.append(a_item)
            if len(cluster) > 1:
                output_boxes.append(
                    {'entity_type': 'cluster',
                     'text': '    '.join([e['text'] for e in cluster]),
                     'box': [
                         min([e['box'][0] - 2 for e in cluster]),
                         min([e['box'][1] - 2 for e in cluster]),
                         max([e['box'][2] + 2 for e in cluster]),
                         max([e['box'][3] + 2 for e in cluster])
                     ]}
                )
        return output, output_boxes

    def process(self, pdf, all_keys):
        full_predictions = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            print("tmpdir: ", tmpdir)
            clean_pdf_path = tmpdir + '/file.pdf'
            with open(pdf, 'rb') as f_in:
                pdf_in = PyPDF2.PdfFileReader(f_in)
                pdf_out = PyPDF2.PdfFileWriter()
                for page in pdf_in.pages:
                    pdf_out.addPage(page)
                pdf_out.removeLinks()
                with open(clean_pdf_path, 'wb') as f_out:
                    pdf_out.write(f_out)
            document = Document(clean_pdf_path)
        boxes = [
            dict(
                page_number=paragraph.page_number,
                text=paragraph.text,
                bbox=paragraph.bbox,
                annots=paragraph.annotation,
            )
            for paragraph in document.paragraphs
        ]
        # process o here
        # full_data = {'data': []}
        min_x = min([float(x['bbox'][0]) for x in boxes])
        min_y = min([float(x['bbox'][1]) for x in boxes])
        max_x = max([float(x['bbox'][2]) for x in boxes])
        max_y = max([float(x['bbox'][3]) for x in boxes])

        pages = sorted(list(set(x['page_number'] for x in boxes)))
        output = {}
        for p in pages:
            if p == 6:
                return output
            _boxes = [box for box in boxes if box['page_number'] == p]
            final_text, bboxes, labels = normalize_boxes(_boxes)
            final_text = unidecode(final_text).lower()

            data = {'title': 'test', 'paragraphs': [{'context': final_text, 'qas': [], "position": bboxes}]}

            for key in all_keys:
                qa = {'question': key,
                      'answers': [],
                      'id': str(uuid4()),
                      'is_impossible': True
                      }
                # else:
                data['paragraphs'][0]['qas'].append(qa)

            predictions = self.model.predict([data], calculate_acc=False, training_mode='NER', test_bs=2)
            full_predictions[p] = predictions

            clustered_items, other_entities = sort_entities(predictions)
            # map back to the original location
            for entity in other_entities:
                # int((box[0] - min_x) / (max_x - min_x) * 1000)
                entity['box'][0] = int(entity['box'][0] / 1000 * (max_x - min_x) + min_x)
                entity['box'][1] = int(entity['box'][1] / 1000 * (max_y - min_y) + min_y)
                entity['box'][2] = int(entity['box'][2] / 1000 * (max_x - min_x) + min_x)
                entity['box'][3] = int(entity['box'][3] / 1000 * (max_y - min_y) + min_y)
            for cluster in clustered_items:
                for entity in cluster:
                    entity['box'][0] = int(entity['box'][0] / 1000 * (max_x - min_x) + min_x)
                    entity['box'][1] = int(entity['box'][1] / 1000 * (max_y - min_y) + min_y)
                    entity['box'][2] = int(entity['box'][2] / 1000 * (max_x - min_x) + min_x)
                    entity['box'][3] = int(entity['box'][3] / 1000 * (max_y - min_y) + min_y)
            json_output, output_boxes = self.write_output(clustered_items, other_entities)
            output[p] = json_output
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(full_predictions, f, ensure_ascii=False)
        return output


if __name__ == '__main__':
    from processors import prj_path

    model_path = prj_path + '/outputs/model_epoch_20'
    question_list_file = prj_path + '/data/question_list.json'
    question_list = json.load(open(question_list_file, 'r', encoding='utf-8'))
    demo = LayoutLMDemo(model_path, version='v1')

    pdf_file = prj_path + '/data/pdf/City Vineyard(second menu).pdf'
    # pdf_file = prj_path + '/data/pdf/testing/Batard.pdf'
    output = demo.process(pdf_file, question_list)
    print(output)
    # json.dump(json_output, open(prj_path + '/data/output.json', 'w', encoding='utf-8'), ensure_ascii=False)
