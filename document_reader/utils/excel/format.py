from functools import lru_cache


class Format:
    @lru_cache(maxsize=16)
    def __init__(self, workbook):
        self.workbook = workbook

        self.default = workbook.add_format()
        self.title = workbook.add_format({'bg_color': '#edf492'})
        self.indexed_formats = {}

        self.red = workbook.add_format({'color': 'red'})
        self.blue = workbook.add_format({'color': 'blue'})
        self.dark_violet = workbook.add_format({'color': '#723ae0'})

        self.bold = workbook.add_format({'bold': True})

        self.bg_red = workbook.add_format({'bg_color': 'red'})

    def __getitem__(self, key):
        if key is None:
            return self.default
        if key in self.indexed_formats.keys():
            return self.indexed_formats[key]
        fmt = self.generate_random_background_color_format()
        self.indexed_formats[key] = fmt
        return fmt

    def generate_random_background_color_format(self):
        def generate_random_bright_color():
            import random
            import colorsys
            h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.6 + random.random() / 4.0
            r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]

            return "#%02X%02X%02X" % (r, g, b)

        return self.workbook.add_format(
            {'bg_color': generate_random_bright_color()})
