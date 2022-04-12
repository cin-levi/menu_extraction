import logging

logger = logging.getLogger(__name__)


class Metric:
    def __init__(self):
        self._scores = []
        self._positive = []
        self._negative = []

    def __add__(self, other):
        metric = Metric()
        metric._scores = self._scores + other._scores
        metric._positive = self._positive + other._positive
        metric._negative = self._negative + other._negative
        return metric

    def add(self, acc: float):
        assert not (self._positive or self._negative), \
            "Use add_positive or add_negative instead"

        self._scores.append(acc)

    def add_positive(self, count: int):
        assert not self._scores, "Use add() instead"

        self._positive.append(count)

    def add_negative(self, count: int):
        assert not self._scores, "Use add() instead"

        self._negative.append(count)

    @property
    def positive(self):
        return sum(self._positive)

    @property
    def negative(self):
        return sum(self._negative)

    @property
    def accuracy(self):
        try:
            if self._scores:
                acc = sum(self._scores) / len(self._scores)
            else:
                acc = self.positive / (self.positive + self.negative)

            return round(100 * acc, 2)
        except BaseException as e:
            # logger.exception(e)
            return 0
