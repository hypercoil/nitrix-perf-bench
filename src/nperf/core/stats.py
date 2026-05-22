# -*- coding: utf-8 -*-
"""Distribution summary for a set of timed samples (L0).

We report the *distribution*, not a point: ``min`` is the best achievable
(least noise-contaminated), ``median`` the typical, ``p95`` the tail.  The
regression gate diffs on ``min`` *and* ``p95`` (DESIGN §L0; the p95 catches
distribution-shape regressions that leave ``min`` untouched).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Distribution:
    '''Summary statistics over timed samples (seconds).'''

    samples: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError('Distribution requires at least one sample.')

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def min(self) -> float:
        return float(np.min(self.samples))

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95))

    @property
    def iqr(self) -> float:
        q75, q25 = np.percentile(self.samples, [75, 25])
        return float(q75 - q25)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    def summary(self) -> Dict[str, float]:
        '''Stored under a metric in the L4 row (numbers only; the metric
        attaches the unit).'''
        return {
            'min': self.min,
            'median': self.median,
            'p95': self.p95,
            'iqr': self.iqr,
            'mean': self.mean,
            'n': self.n,
        }

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> 'Distribution':
        return cls(tuple(float(s) for s in samples))
