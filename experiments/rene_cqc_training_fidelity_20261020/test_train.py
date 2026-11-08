import unittest
from copy import deepcopy

from .common import ARM_SPECS, lr_at


class TrainingContract(unittest.TestCase):
    def test_single_changes(self):
        keys={'O':{'optimizer'},'S':{'schedule','horizon'},'N':{'max_length'},'W':{'weight_decay'}}
        for name,expected in keys.items():
            self.assertEqual({k for k in ARM_SPECS['L'] if ARM_SPECS['L'][k]!=ARM_SPECS[name][k]},expected)

    def test_literal_schedule_does_not_decay_to_zero(self):
        self.assertAlmostEqual(lr_at(ARM_SPECS['S'],1024),.001*(1-1024/100000))
        self.assertGreater(lr_at(ARM_SPECS['S'],1024),.00098)

    def test_scaled_schedule_and_seed_controls(self):
        self.assertEqual(ARM_SPECS['PN']['horizon'],1093)
        self.assertEqual({k for k in ARM_SPECS['P'] if ARM_SPECS['P'][k]!=ARM_SPECS['PS'][k]},{'seed'})
        self.assertGreater(lr_at(ARM_SPECS['PN'],1024),0)
        self.assertLess(lr_at(ARM_SPECS['PN'],1024),.000065)

    def test_length_cap_identical_when_below_limit(self):
        # Scientific null: same dynamic padding and no truncation gives identical tensors, irrespective of declared cap.
        import torch
        from transformers import AutoTokenizer
        from .common import BASE
        from .train import encode
        tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        bank=[dict(source='get <extra_id_0> name',target='<extra_id_0> user <extra_id_1>'),
              dict(source='load <extra_id_0>',target='<extra_id_0> file <extra_id_1>')]
        a,la=encode(tokenizer,bank,128,'cpu');b,lb=encode(tokenizer,bank,512,'cpu')
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a));self.assertTrue(torch.equal(la,lb))


if __name__=='__main__':unittest.main()
