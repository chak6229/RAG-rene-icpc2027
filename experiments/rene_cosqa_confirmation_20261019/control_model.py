"""Matched-supervision control: the same update model receives no CE columns."""
from __future__ import annotations


class NoCEPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, matrix):
        return self.model.predict(matrix[:, :73])
