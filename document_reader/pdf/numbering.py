import logging
import re

from ..numbering import Numbering as BaseNumbering

logger = logging.getLogger(__name__)


class Numbering(BaseNumbering):

    def __init__(self, text: str):
        self.text = _parse(text)

    @property
    def text(self) -> str:
        return self._text or ''

    @text.setter
    def text(self, value: str):
        self._text = value

    def is_subsequence_of(self, other):
        if not other:
            return False

        try:
            index_pattern = re.compile(r'\d+')

            this_text = index_pattern.sub('', self.text, count=1)
            other_text = index_pattern.sub('', other.text, count=1)

            if this_text == other_text:
                return self._is_valid_subsequence(index_pattern, other)
        except Exception as e:
            logger.warning(f'Numbering::is_subsequence_of! {e}')

        return False

    def is_subsection_of(self, other):
        if not other:
            return False

        other_text: str = other.text
        if other_text.endswith('.0.'):
            other_text = other_text[:-2]

        if 0 < len(self.text) - len(other_text) <= 2:
            r = self.text.startswith(other_text)
            if r:
                if other.text == '2':
                    return len(self.text) - len(other.text) > 1
                return True

        return False

    def is_initial(self):
        matches = re.findall(r'\d', self.text)
        return len(matches) == 1 and matches[0] == '1'

    def _is_valid_subsequence(self, index_pattern, other):
        this_index = index_pattern.findall(self.text)
        other_index = index_pattern.findall(other.text)

        if len(this_index) != len(other_index):
            logger.warning(f'Different base length: '
                           f'{this_index} vs. {other_index}')
            return False

        this_index = int(this_index[-1])
        other_index = int(other_index[-1])

        return this_index == (other_index + 1)


def _parse(text):
    from ..utils.utils import normalize_text
    text = normalize_text(text, remove_space=False)
    from ..utils.config import settings
    for pattern in settings.pdf.numbering.patterns:
        matches = re.findall(pattern, text + ' ')
        if len(matches) > 0:
            text = matches[0]
            return text.strip().rstrip(re.sub(r'\d+', '', text))
