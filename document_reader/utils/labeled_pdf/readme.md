Labeled PDF Tool
---
To convert labeled PDF to the standard output xlsx

A PDF can be labeled by annotating on the file directly. The annotation must
follow [this convention](https://drive.google.com/file/d/11FYn9XwAJ5JHLr88sfdllb3GaKJhiEI6/view?usp=sharing)

Here's [an example](https://drive.google.com/file/d/1KC4TuEzM0OLL8QNeYYdZWi8stbpm0lpE/view?usp=sharing)

Usage
---

### Basic usage

```python
from document_reader import LabeledPDF

pdf = LabeledPDF('path/to/file.pdf')

# export the labeled pdf to xlsx format, 
# the xlsx shares the same name and directory as the pdf 
pdf.to_xlsx()
# or specify the output path
pdf.to_xlsx(out_dir='path/to/output_file.xlsx')
```

There are two types of label:

- for tags
- for structure of the document

Label for tags is mandatory, and label for structure is optional.

### Dry-run test

```python
from document_reader import validate_labeled_pdf

path = 'path/to/file.pdf'
try:
    validate_labeled_pdf(path)
except AssertionError as e:
    # To detail on the validation fail
    print(f'{e}')
```

The above could possibly print out:

```text
L01: Corrupted or not a PDF
L02: The PDF does not have any text
L03: Missing annotations
L04: Unsupported language. PDF must be in Japanese or English
L05: Cannot parse annotation: <detail...>

L00: Unclassified error: "<detail...>"
```

If the pdf pass the validation, it's expected to be able to be exported to xlsx.

