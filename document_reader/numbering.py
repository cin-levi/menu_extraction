class Numbering:
    def __bool__(self):
        return bool(self.text)

    def __repr__(self):
        return self.text

    @property
    def text(self) -> str:
        return ''

    def is_subsequence_of(self, other) -> bool:
        """ Same as 1.2(self) to 1.1(other)
        """
        raise NotImplementedError()

    def is_subsection_of(self, other) -> bool:
        """ Same as of 1.1.1(self) to 1.1(other)
        """
        raise NotImplementedError()

    def is_initial(self) -> bool:
        """ True if this numbering is the first item of the hierarchy.
        Example: `1.`, `(a)`, or `I`
        """
        raise NotImplementedError()
