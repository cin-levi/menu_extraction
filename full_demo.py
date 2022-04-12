import json, tempfile, PyPDF2

from document_reader import Document
from models.inference import LayoutLMInference, LayoutLMV2Inference
from unidecode import unidecode
from uuid import uuid4
from utils.convert_invoice_to_qa_format import normalize_boxes
from models.sort_entities import sort_parts

class LayoutLMDemo(object):
    def __init__(self, model_path, version='v1'):
        if version == 'v1':
            self.model = LayoutLMInference.load_from_checkpoint(model_path)
        else:
            self.model = LayoutLMV2Inference.load_from_checkpoint(model_path)
        self.label_dict = {'b': 'brand',
                           'ba': 'brand_address',
                           'pname': 'payment_name',
                           'pnum': 'payment_number'}
        pass

    def sort_items(self, output):
        accepted_entities = []
        other_entities = []
        for key in output:
            for entity in output[key]:
                entity['box'] = [min([x[0] for x in entity['position']]),
                                 min([x[1] for x in entity['position']]),
                                 max([x[2] for x in entity['position']]),
                                 max([x[3] for x in entity['position']]),
                                 ]
            accepted_entities.extend([x for x in output[key] if x['entity_type'] in ['v', 'w', 'p']])
            other_entities.extend([x for x in output[key] if x['entity_type'] not in ['v', 'w', 'p']])
        #TODO: The algorithm need to be changed a bit here
        # Use dbscan to detect the
        clustered_items = sort_parts(accepted_entities)[0]
        return clustered_items, other_entities

    def write_output(self, clustered_items, other_entities):
        output_boxes = []
        output = {'items': []}
        for e in other_entities:
            if e['entity_type'] in self.label_dict:
                label = self.label_dict[e['entity_type']]
            else:
                label = e['entity_type']
            if label not in output:
                output[label] = ''
            output[label] += ' ' + e['text']
            output_boxes.append(e)
            print(e)
        for cluster in clustered_items:
            print(cluster)
            a_item = {}
            for e in cluster:

                if e['entity_type'] != 'item_number':
                    if e['entity_type'] in self.label_dict:
                        label = self.label_dict[e['entity_type']]
                    else:
                        label = e['entity_type']
                    a_item[label] = e['text']
                else:
                    a_item['quantity'] = e['text']
                output_boxes.append(e)
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
            output['items'].append(a_item)
        return output, output_boxes

    def process(self, pdf, all_keys):
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
            if p == 5:
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

            full_data= {'data': data}
            predictions = self.model.predict(full_data['data'], calculate_acc=False, training_mode='NER', test_bs=2)
            clustered_items, other_entities = self.sort_items(predictions)
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
        return output


if __name__ == '__main__':
    from processors import prj_path

    model_path = prj_path + '/outputs/model_epoch_30'
    question_list_file = prj_path + '/data/wine_menus/question_list.json'
    question_list = json.load(open(question_list_file, 'r', encoding='utf-8'))
    demo = LayoutLMDemo(model_path, version='v1')

    pdf_file = prj_path + '/data/pdf/testing/Adda - wine reserve.pdf'
    # pdf_file = prj_path + '/data/pdf/testing/Batard.pdf'
    output = demo.process(pdf_file, question_list)
    print(output)
    # json.dump(json_output, open(prj_path + '/data/output.json', 'w', encoding='utf-8'), ensure_ascii=False)
