from processors import prj_path
import json
from full_demo import LayoutLMDemo
from pydantic import BaseModel
from flask import Flask, request, jsonify

app = Flask('levi_invoice_demo')

model_path = prj_path + '/outputs/model_epoch_30'
question_list_file = prj_path + '/data/question_list.json'
question_list = json.load(open(question_list_file, 'r', encoding='utf-8'))
demo = LayoutLMDemo(model_path)


class JsonData(BaseModel):
    file: str


@app.route("/inference", methods=['POST'])
def inference():
    try:
        file = request.files.get("file")
        with open("tmp.pdf", 'wb') as f:
            f.write(file.read())

        # Process
        json_output = demo.process("tmp.pdf", question_list)
    except Exception as e:
        json_output = {'error': e}

    return jsonify(json_output)


if __name__ == '__main__':
    app.run('0.0.0.0', port=8000)
