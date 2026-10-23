import unittest

from complete_api_coverage import Budget, find_missing


class CoverageTests(unittest.TestCase):
    def test_never_reruns_local_or_existing_stages(self):
        base = dict(model='Haiku-5.5', sample_id='1', stage='baseline',
                    label='SUPPORTS', ground_truth_label='SUPPORTS', needs_review='False')
        rows = [base, {**base, 'stage': 'weak'}, {**base, 'model': 'GPT-OSS-20B'},
                {**base, 'sample_id': '2', 'needs_review': 'True'}]
        jobs = find_missing(rows, {('Haiku-5.5', '1', 'strong')})
        self.assertEqual([(row['sample_id'], stage) for row, stage in jobs], [('1', 'medium')])

    def test_budget_reserves_inflight_maxima(self):
        budget = Budget(.1)
        budget.reserve(.06)
        with self.assertRaises(RuntimeError):
            budget.reserve(.06)
        budget.settle(.06, .001)
        budget.reserve(.06)
        self.assertAlmostEqual(budget.spent, .001)


if __name__ == '__main__':
    unittest.main()
