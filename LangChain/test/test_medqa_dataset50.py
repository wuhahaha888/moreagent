"""Protect the paid experiment's fixed stream and evaluator/input boundary."""
from collections import Counter
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from ..medqa.data import Dataset

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'experiments/datasets/medical-related-50-v2'


class Dataset50Test(unittest.TestCase):
    def test_original_gold_and_order_unchanged_and_new_edges_acyclic(self):
        old=HERE.parent/'medical-related-30-v1'
        original=json.loads((old/'runner-benchmark.json').read_text())['medqa']
        order=json.loads((HERE/'order.json').read_text())['ids']
        runner=json.loads((HERE/'runner-benchmark.json').read_text())['medqa']
        self.assertEqual(order[:30],json.loads((old/'order.json').read_text())['ids'])
        self.assertEqual({k:runner[k] for k in original},original)
        self.assertEqual(len(set(order)),50)
        reviews=json.loads((HERE/'evaluation/reuse-review.json').read_text())['tasks']
        for r in reviews:
            self.assertTrue(set(r['reuse_from_task_ids'])<=set(order[:r['global_position']-1]))
        self.assertEqual(Counter(r['group_id'] for r in reviews),{'anemia':7,'renal':7,'thyroid':6})
        self.assertEqual(Counter(runner[k]['answer'] for k in order[30:]),dict.fromkeys('ABCD',5))

    def test_agent_inputs_match_public_questions_and_exclude_evaluation_metadata(self):
        data=Dataset(HERE/'runner-benchmark.json')
        public=[json.loads(s) for s in (HERE/'questions.jsonl').read_text().splitlines()]
        self.assertEqual([q.public() for q in data.select(split='all')],public)
        for q in public:
            self.assertEqual(set(q),{'question_id','question','options'})
            self.assertEqual(set(q['options']),set('ABCD'))

    def test_offline_rebuild_is_reproducible(self):
        build=runpy.run_path(str(HERE/'build.py'))['build']
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);build(output)
            for name in json.loads((HERE/'checksums.json').read_text()):
                if (output/name).is_file():
                    self.assertEqual((output/name).read_bytes(),(HERE/name).read_bytes(),name)


if __name__=='__main__':unittest.main()
