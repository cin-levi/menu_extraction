Excel Utility
---

A thin wrapper for xlsxwriter to make it more convenient

Sample usage:
```python
from document_reader.utils.excel import Excel

with Excel('path_to_file.xlsx') as excel:
    sheet, formats = excel.add_sheet(
        ['Name', 'Age'],  # column names
        [50, 10],  # column widths
    )

    sheet.write_row('John Doe', 22)
    sheet.write_row('Fiona Smith', 20)

    # add format to a cell by making a tuple
    sheet.write_row(('Ada Lovelace', formats.red), 23)

    # write at a specific location
    # change the age for Fiona to 21
    sheet.write(21, row_index=2, col_index=1)
```
