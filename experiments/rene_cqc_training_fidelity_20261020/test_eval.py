import itertools
import math
import unittest
from fractions import Fraction

from .analyze import exact_max
from .audit import cdf_max
from .evaluate import order


class ScientificEvaluation(unittest.TestCase):
    def test_exact_random_duplicate_positions(self):
        values=[Fraction(0),Fraction(1,2),Fraction(1,2),Fraction(1)]
        brute=sum((max(v) for v in itertools.combinations(values,3)),Fraction(0))/4
        self.assertEqual(cdf_max(values),brute)
        self.assertAlmostEqual(exact_max([float(v) for v in values]),float(brute))

    def test_short_pool_keeps_all_real_positions(self):
        self.assertEqual(cdf_max([Fraction(1,3),Fraction(1,2)]),Fraction(1,2))
        self.assertEqual(exact_max([1/3,1/2]),.5)

    def test_order_tie_break_and_probability_direction(self):
        pool=[dict(position=2,full_vocab_entropy=1.,label_mean_log_probability=-1.),
              dict(position=1,full_vocab_entropy=1.,label_mean_log_probability=-2.)]
        self.assertEqual(order(pool,'ENTR'),[1,2]);self.assertEqual(order(pool,'PROB'),[2,1])

    def test_single_factor_and_primary_size(self):
        from .common import ARM_SPECS
        self.assertEqual((len(ARM_SPECS)-1)*2*2,28)


if __name__=='__main__':unittest.main()
