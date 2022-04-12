from collections import defaultdict

from xlsxwriter.format import Format

from .utils import index_generator


class Sheet:
    def __init__(self, sheet):
        self.sheet = sheet
        self._col_indices = defaultdict(index_generator)
        self._row_indices = index_generator(1)

    def write(self, *args, row_index=None, col_index=None):
        """ Write data to a specific cell
        :param args: the data to be writen to the cell
        :param row_index: index of the row to be writen, default: next to the last writen row
        :param col_index: index of the column to be writen, default: next to the last column on the row
        """
        row_index = row_index or next(self._row_indices)
        col_index = col_index or next(self._col_indices[row_index])

        if len(args) in (1, 2):
            self.sheet.write(
                row_index, col_index,
                *(reversed(args) if isinstance(args[0], Format) else args))
        elif args:
            self.sheet.write_rich_string(row_index, col_index, *args)

    def write_row(self, *args):
        """ Append a full row with provided data
        :param args: list of cell data to be written if *args format, rich text
        can be added by using tuple
        """
        # self.sheet.write_row(next(self._row_indices), 0, args)
        index = next(self._row_indices)
        for arg in args:
            if isinstance(arg, tuple):
                self.write(*arg, row_index=index)
            else:
                self.write(arg, row_index=index)
