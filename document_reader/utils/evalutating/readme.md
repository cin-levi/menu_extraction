Evaluating
---
Providing utilities to evaluate document reader output against the available ground truth(CA). There
are three stages involve in the process of document reader, namely:

- Stage 1: Getting the characters from the PDF
- Stage 2: Constructing paragraphs from the characters
- Stage 3: Building a tree representing the structure of the document. So we will evaluate each of
  these stages. Any error happen in previous stage will directly affect the outcome of the next
  stage. It needed to be mitigated so that we can get more relevant metric -> introducing correcting
  step after each stage

Usage
---

To create accuracy report

```python
from document_reader.utils import evalutating

evalutating.gen_report(
  dr='path_to_document_reader_output',
  ca='path_to_ground_truth'
)
# You can also specify the output directory, and the number of jobs to run in parallel. 
# See the function docstring for detail

# Or use short-handed utility
from document_reader.utils import dr

dr.gen_report(ca_dir='path_to_ground_truth')
# This will collect the output for document reader in default output location 
# specified by the settings module
```

Output for `gen_report` will be an excel file for each input file, will be stored under `.report` by
default. The Excel has four sheets, corresponding to each of the three stages above, and one
additional sheet for the overall metrics.

In addition to `gen_report`, there are more utilities:

- `gen_dictionaary` for generating dictionary for postprocessing OCR
- `gen_diff_as_html` for generating unified diff for doc reader output and the ground truth
- `save_excel_as_text` for saving the content of the document representing in Excel file as plain
  text For usage, see the docstring for each function.
