# Wine menu extraction


## General information

### Back bone

This invoice extraction based on:

- Auto-encoder LM (Denoising Language model) pretrained model: BERT, RoBERTa, LayoutLM, LayoutLMv2 (next update)
- PdfPlumber
- Levi 's QA architectures ([Normal QA](https://ieeexplore.ieee.org/document/9534328)
  , [Recusive QA - trial version](https://arxiv.org/pdf/2106.00978.pdf))
- Supported languages: English
- Model type: NER, QA, Hybrid. *Please ask levi for the explanation for use case of each type*

### Overall architecture

The system architecture is
described [here](https://www.notion.so/scalr/Research-Notes-485f86a8a0d04d1588a401b35631291d?p=228ac50a1699416c86ce3c073d062af6&showMoveTo=true)
privately for Width.ai members.

*THIS ARCHITECTURE IS STRICTLY CONFIDENTIAL*

## Training AI model

### Data format

The training file is similar to [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) with some important
modification.

Training folder:

```bash
|----data
|     |----dataset_1
|     |       |----train.json
|     |       |----test.json
|     |       |----question_list.json

```

Example: `data/Cord/`

Format:
```json
{
  "data": [
    {
      "title": "invoice-10.pdf.json",
      "paragraphs": [
        {
          "context": "invoice cellphones 0344334446 hai ba trung cell@gmail.com ha noi cell.com bill to ship to invoice details pham van manh pham huy hung invoice #: 1423482423 truong chinh hai ba trung invoice date: 02/07/2020 ha noi pham hung terms: due: 02/08/2022 product/service description qty/hrs rate amount 1 macbook pro 20000000 1 20,000,000.00 2 air pod pro 6000000 1 6,000,000.00 subtotal 26,000,000.00 sales tax 55000 shipping total $26,055,000.00 due 02/08/2022",
          "qas": [
            {
              "question": "Company",
              "answers": [
                {
                  "answer_start": 8,
                  "text": "cellphones"
                }
              ]
            }
          ],
          "position": [
            [
              30,
              0,
              172,
              39
            ],
            [
              30,
              94,
              55,
              114
            ]
          ]
        }
      ]
    }
  ]
}
```


Notice:

- Each document of `train.json` and `test.json` have to contain all questions listed in `question_list.json`
- Every answer have `is_impossible=False` have to contain at least one sample with no-blank answer
- This condition must be hold: `len(paragraph["position"]) == len(paragraph["context"].split())`
- The position is in `[x0, y0, x1, y1]` (integer) format and scaled to `[0, 1000]`

### Train model
To reproduce the experiments, run the following command lines:
```
export PYTHONPATH=.
mkdir caches
mkdir outputs
CUDA_VISIBLE_DEVICES=0 python experimens/train_invoice.py
```
See `experimens/train_invoice.py` for more detail

## Full-flow process
```python
from full_demo import LayoutLMDemo
from processors import prj_path

model_path = prj_path + '/outputs/invoices/model_epoch_20' # path to trained model

demo = LayoutLMDemo(model_path)

pdf_file = prj_path + '/data/test.pdf' # path to invoice PDF file

predictions = demo.process(pdf_file)
```

## Important modules
Support modules:
- `experiments/test_document_reader.py`: Example of reading annotated data from PDF files to json files
- `data/invoice/convert_invoice_to_qa_format.py`: Convert json files to trainable data
- `utils/convert_single_answer_to_multi_answer.py`: Add missing questions
- `experiments/train_invoice.py`: Example of training LayoutLM for Invoice dataset

Main modules:
- `processors/*.py`: scrips for processing data
- `models/*.py`: main model architecture, trainer and optimizer
- `utils/*.py`: support functions

For more information, please ask Levi (`nguyenhongson.kstn.hust@gmail.com`)