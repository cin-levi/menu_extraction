from typing import List, Tuple

import xlsxwriter

from .format import Format
from .sheet import Sheet


class Excel(xlsxwriter.Workbook):
    def add_sheet(self, headers: List[str], col_widths: List[int], name: str = 'Sheet1') -> Tuple[Sheet, Format]:
        """ Setup sheet with provided header and column widths.
        The sheet is named 'Sheet1' and has a frozen header row.
        :param self: the workbook where sheet will be created.
        :param headers: column headers of the sheet.
        :param col_widths: the widths of each column in the sheet.
        :param name: name of the sheet to be created, default: `Sheet1`.
        :return: the sheet and the format.
        """
        sheet = self.add_worksheet(name)
        header_format = self.add_format(dict(align='center', bold=True))

        for index, headers in enumerate(headers):
            sheet.write_string(0, index, headers, header_format)

        sheet.freeze_panes(1, 0)

        for i, width in enumerate(col_widths):
            sheet.set_column(i, i, width)

        return Sheet(sheet), self.get_format()

    def get_format(self):
        return Format(self)
