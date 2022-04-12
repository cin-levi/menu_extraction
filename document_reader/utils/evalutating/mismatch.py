class Mismatch:
    def __init__(self, prefix: str, suffix: str,
                 text_del: str, text_ins: str, ca_name: str):
        self.text_del = text_del
        self.text_ins = text_ins
        self.suffix = suffix
        self.prefix = prefix
        self.ca_name = ca_name

    def __lt__(self, other):
        return self._signature() < other._signature()

    def __eq__(self, other):
        return self._signature() == other._signature()

    def __hash__(self):
        return hash(self._signature())

    @property
    def text1(self):
        return ''.join((self.prefix, self.text_del, self.suffix)).strip()

    @property
    def text2(self):
        return ''.join((self.prefix, self.text_ins, self.suffix)).strip()

    def _signature(self):
        return f'{self.text_del} {self.text_ins} {self.prefix} {self.suffix}'
