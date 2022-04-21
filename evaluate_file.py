import json, tempfile, PyPDF2

from document_reader import Document
from models.inference import LayoutLMInference, LayoutLMV2Inference
from unidecode import unidecode
from uuid import uuid4
from utils.convert_invoice_to_qa_format import normalize_boxes
from models.sort_entities import sort_entities


class LayoutLMEvaluate(object):
    def __init__(self, model_path, version='v1'):
        self.model = LayoutLMInference.load_from_checkpoint(model_path)



    def process(self, json_data):
        full_predictions = {}
        for sample in json_data:
            title = sample['title']
            predictions = self.model.predict([sample], calculate_acc=False, training_mode='NER', test_bs=2)
            print(predictions)
            print(sample)
            1/0


if __name__ == '__main__':
    from processors import prj_path

    model_path = prj_path + '/outputs/model_epoch_20'
    question_list_file = prj_path + '/data/question_list.json'
    question_list = json.load(open(question_list_file, 'r', encoding='utf-8'))
    demo = LayoutLMDemo(model_path, version='v1')

    # pdf_file = prj_path + '/data/pdf/Adda - wine reserve.pdf'
    pdf_file = prj_path + '/data/pdf/45. Contento.pdf'
    # pdf_file = prj_path + '/data/pdf/24. Gage _ Tollner.pdf'
    # pdf_file = prj_path + '/data/pdf/testing/Batard.pdf'
    file = 'D:\\menu_extraction\\data\\full.json'
    json_data = json.load(open(file, 'r', encoding='utf-8'))['data']
    output = demo.process(json_data)
    print(output)
    # json.dump(json_output, open(prj_path + '/data/output.json', 'w', encoding='utf-8'), ensure_ascii=False)
