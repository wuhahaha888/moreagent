"""并行入口的顺序、隔离和续跑测试；使用替身，不发起模型请求。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .test_medqa import Agents, Books
from ..medqa import cli
from ..medqa.data import Dataset
from ..medqa.runtime import Runtime

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('parallel_runner_test', ROOT / 'scripts/run_medqa_parallel.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ParallelTest(unittest.TestCase):
    def config(self, root):
        benchmark = root / 'benchmark.json'
        benchmark.write_text(json.dumps({'medqa': {key: {'question': 'fixture',
            'options': {k: k for k in 'ABCD'}, 'answer': 'A'} for key in ['q1', 'q2']}}))
        data = Dataset(benchmark)
        ids = root / 'ids.json'
        ids.write_text(json.dumps({'dataset_sha256': data.sha256, 'ids': ['q2', 'q1']}))
        config = root / 'config.json'
        config.write_text(json.dumps({'benchmark': str(benchmark), 'ids': str(ids), 'split': 'all',
            'output': str(root / 'result'), 'model_name': 'qwen3-14b-bf16',
            'model_base_url': 'http://127.0.0.1:8010/v1'}))
        return config, root / 'result'

    def test_four_workers_keep_order_isolate_outputs_and_skip_completed_on_resume(self):
        with tempfile.TemporaryDirectory() as name:
            config, out = self.config(Path(name))
            launched = []
            class Worker:
                def __init__(self, command, **kwargs):
                    cfg = json.loads(Path(command[command.index('--config') + 1]).read_text())
                    launched.append(cfg)
                    self.pid = 1000 + len(launched)
                    folder = Path(cfg['output']);folder.mkdir(parents=True)
                    condition = cfg['mode'] + '_' + cfg['memory']
                    data = Dataset(Path(cfg['benchmark']))
                    rows = []
                    for question in data.select(ids=['q2', 'q1'], split='all'):
                        result = Runtime(Agents(), Books()).run(question)
                        result['correct'] = True
                        rows.append(result)
                    cli.dump(folder / 'report.json', {condition: cli.summarize(rows)})
                    cli.dump(folder / 'manifest.json', {'question_ids': ['q2', 'q1']})
                    (folder / (condition + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
                def poll(self):
                    return 0
            argv = ['parallel', '--config', str(config), '--all', '--workers', '4']
            with patch('sys.argv', argv), patch.object(runner, 'check_local_service'), patch.object(runner.subprocess, 'Popen', Worker), patch.object(runner.time, 'sleep'):
                runner.main()
            self.assertEqual(len(launched), 4)
            self.assertEqual(len({c['output'] for c in launched}), 4)
            self.assertEqual({(c['mode'], c['memory']) for c in launched},
                             {('text', 'off'), ('structured', 'off'), ('text', 'on'), ('structured', 'on')})
            for c in runner.CONDITIONS:
                rows = [json.loads(line) for line in (out / (c + '.jsonl')).read_text().splitlines()]
                self.assertEqual([r['question_id'] for r in rows], ['q2', 'q1'])
            comparison = json.loads((out / 'comparison.json').read_text())
            self.assertTrue(comparison['matched_complete'])
            self.assertFalse(comparison['latency_comparison_valid'])
            self.assertEqual(sum(m['model_calls'] for m in json.loads((out / 'report.json').read_text()).values()), 24)
            with patch('sys.argv', argv + ['--resume']), patch.object(runner, 'check_local_service'), patch.object(runner.subprocess, 'Popen') as popen:
                runner.main()
                popen.assert_not_called()

    def test_dry_run_makes_no_worker_and_changed_concurrency_cannot_resume(self):
        with tempfile.TemporaryDirectory() as name:
            config, out = self.config(Path(name))
            argv = ['parallel', '--config', str(config), '--all', '--dry-run']
            with patch('sys.argv', argv), patch.object(runner.subprocess, 'Popen') as popen:
                runner.main()
                popen.assert_not_called()
            self.assertFalse((out / 'conditions/text_on').exists())
            with patch('sys.argv', argv + ['--resume', '--workers', '2']), self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)

    def test_unavailable_service_creates_no_experiment_and_starts_no_worker(self):
        with tempfile.TemporaryDirectory() as name:
            config, out = self.config(Path(name))
            with patch('sys.argv', ['parallel', '--config', str(config), '--all']), \
                 patch.object(runner, 'check_local_service', side_effect=SystemExit('service unavailable')), \
                 patch.object(runner.subprocess, 'Popen') as popen:
                with self.assertRaises(SystemExit):
                    runner.main()
                popen.assert_not_called()
            self.assertFalse(out.exists())

    def test_service_probe_rejects_wrong_model_without_generation(self):
        from io import BytesIO
        from unittest.mock import MagicMock
        opener = MagicMock()
        response = BytesIO(b'{"data":[{"id":"another-model"}]}')
        opener.open.return_value.__enter__.return_value = response
        with patch('urllib.request.build_opener', return_value=opener), self.assertRaises(SystemExit):
            runner.check_local_service('qwen3-14b-bf16', 'http://127.0.0.1:8010/v1', 'local')
        request = opener.open.call_args.args[0]
        self.assertTrue(request.full_url.endswith('/models'))
        self.assertIsNone(request.data)


if __name__ == '__main__':
    unittest.main()
