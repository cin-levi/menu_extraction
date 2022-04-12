class Bullet:
    def __bool__(self):
        return bool(self.text)

    def __repr__(self):
        return self.text

    @property
    def text(self) -> str:
        return ''
