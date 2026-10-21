"""Workspace tests exercise actual prompts, transport, memory and crash replay."""
from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .test_medqa import Books, Embeddings
from .test_medqa_communication import ScriptedChat
from .test_medqa_checkpoint import PowerCut
from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from ..medqa import cli
from ..medqa.checkpoint import DurableTools
from ..medqa.data import Question
from ..medqa.grounding import ContractError
from ..medqa.memory import SharedMemory
from ..medqa.runtime import Runtime, ChatRoles
from ..medqa.workspace import Workspace, RoleViewBuilder, brief_is_eligible, POLICY


class Model:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        payload = json.loads(messages[1].content)
        prompt = messages[0].content
        phase = 0 if '你识别当前需求' in prompt else 1 if '你负责有边界的检索策略' in prompt else 2
        script = ScriptedChat()
        script.messages = [None] * phase
        response = script.invoke(messages)
        data = json.loads(response.content)
        data['question_id'] = payload['question_id']
        if phase == 0:
            selected = [h for h in payload.get('memory_hits', []) if h.get('kind') == 'evidence'][:1]
            data['memory_assessments'] = [{'memory_id': h['id'], 'knowledge_task_ids': ['K1']} for h in selected]
        response.content = json.dumps(data, ensure_ascii=False)
        return response


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.q = Question('q', 'Synthetic NOT condition, adult, 10 mg, time limit.', {k: k for k in 'ABCD'})

    def run_case(self, model=None, **kwargs):
        model = model or Model()
        agents = ChatRoles(model, output_policy='decoupled', memory_selection='needs', workspace_policy=POLICY)
        runtime = Runtime(agents, Books(), output_policy='decoupled', memory_policy='basic',
                          memory_selection='needs', workspace_policy=POLICY, max_calls=4, **kwargs)
        return runtime.run(self.q), model, runtime

    def test_invalid_patch_is_atomic_and_cannot_change_original_task(self):
        state = {}; original = self.q.public(); workspace = Workspace(original, state)
        original['options']['A'] = 'tampered'
        workspace.task['options']['A'] = 'tampered'
        self.assertEqual(workspace.task['options']['A'], 'A')
        for patch_value in ({'question': 'changed'}, {'criteria': ['valid'], 'options': {}}, {'criteria': [float('nan')]}):
            before = deepcopy(state)
            with self.assertRaises((ContractError, ValueError)):
                workspace.apply(PLANNER, patch_value)
            self.assertEqual(state, before)
            self.assertEqual(workspace.revision, 0)
        workspace.apply(PLANNER, {'criteria': ['source support']})
        self.assertEqual(workspace.snapshot()['revision'], 1)

    def test_brief_requires_source_quotes_numbers_and_negated_sentence(self):
        question = 'An adult has 10 mg exposure. No prior exposure. What mechanism explains this?'
        conditions = [{'id': 'C1', 'text': 'An adult has 10 mg exposure.'}, {'id': 'C2', 'text': 'No prior exposure.'}]
        target = 'What mechanism explains this?'
        self.assertTrue(brief_is_eligible(question, conditions, target))
        self.assertFalse(brief_is_eligible(question, conditions[:1], target))
        self.assertFalse(brief_is_eligible(question, conditions[1:], target))
        self.assertFalse(brief_is_eligible(question, [{'id':'C1','text':'an inferred condition'}], target))

    def test_source_brief_is_optional_and_all_options_are_preserved(self):
        question = ('An extensive historical narrative describes prior routine observations. ' * 30
                    + 'An adult has 10 mg exposure. What mechanism explains this?')
        payload = {**self.q.public(), 'question': question,
                   'key_conditions': [{'id':'C1','text':'An adult has 10 mg exposure.'}],
                   'knowledge_tasks': [{'id':'K1','goal':'check mechanism','condition_ids':['C1']}], 'budget':{}}
        view, audit = RoleViewBuilder().build(INVESTIGATOR, payload)
        self.assertEqual(audit['task_view'], 'source_quoted_brief')
        self.assertEqual(view['options'], self.q.options)
        payload['key_conditions'][0]['text'] = 'translated, unanchored summary'
        fallback, audit = RoleViewBuilder().build(INVESTIGATOR, payload)
        self.assertEqual(fallback['question'], question)
        self.assertEqual(audit['task_view'], 'full_task_without_duplicate_conditions')

    def test_actual_model_input_is_projected_and_acp_does_not_reexpand_it(self):
        result, model, runtime = self.run_case()
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(result['metrics']['model_calls'], 3)
        self.assertEqual(result['metrics']['workspace_patch_count'], 3)
        events = [e for e in result['events'] if 'model_input' in e]
        for event, messages in zip(events, model.messages):
            self.assertEqual(event['model_input'], messages[1].content)
        investigator = json.loads(model.messages[1][1].content)
        reasoner = json.loads(model.messages[2][1].content)
        self.assertNotIn('key_conditions', investigator)
        self.assertNotIn('task_status', investigator)
        self.assertNotIn('key_conditions', reasoner)
        self.assertNotIn('memory_hits', reasoner)
        self.assertEqual(reasoner['question'], self.q.question)
        self.assertEqual(reasoner['options'], self.q.options)
        self.assertEqual(reasoner['evidence_pool'][0]['text'], Books().doc['text'])
        self.assertTrue(runtime.workspace.state['key_conditions'])
        self.assertTrue(runtime.workspace.state['investigation_plan'])
        self.assertEqual(runtime.workspace.state['reasoning_result']['answer'], 'A')
        self.assertTrue(result['metrics']['workspace_input_bytes_after'] < result['metrics']['workspace_input_bytes_before'])

    def test_text_baseline_has_identical_inputs_and_prompts(self):
        captures = []
        for policy in ('legacy', POLICY):
            model = ScriptedChat()
            agents = ChatRoles(model, output_policy='decoupled', memory_selection='needs', workspace_policy=policy)
            result = Runtime(agents, Books(), mode='text', output_policy='decoupled', memory_policy='basic',
                             memory_selection='needs', workspace_policy=policy).run(self.q)
            self.assertEqual(result['answer'], 'A')
            self.assertEqual(result['metrics']['workspace_views'], 0)
            self.assertIsNone(result['workspace_version'])
            captures.append([[m.content for m in messages] for messages in model.messages])
        self.assertEqual(captures[0], captures[1])

    def test_acp_transports_a_brief_without_inserting_the_original_stem_again(self):
        class Quoted(Model):
            def invoke(self, messages):
                response = super().invoke(messages)
                data = json.loads(response.content)
                if 'key_conditions' in data:
                    data['key_conditions'] = [{'id':'C1','text':'An adult has 10 mg exposure.'}]
                response.content = json.dumps(data)
                return response
        self.q = Question('q', 'An extensive historical narrative describes prior routine observations. '*30
                          + 'An adult has 10 mg exposure. What mechanism explains this?', self.q.options)
        result, model, _ = self.run_case(model=Quoted())
        self.assertEqual(result['answer'], 'A')
        investigator = json.loads(model.messages[1][1].content)
        reasoner = json.loads(model.messages[2][1].content)
        self.assertEqual(investigator['question'], 'What mechanism explains this?')
        self.assertEqual(investigator['options'], self.q.options)
        self.assertEqual(reasoner['question'], self.q.question)
        self.assertEqual(result['metrics']['workspace_brief_views'], 1)

    def test_optional_followup_keeps_new_source_text_visible_and_counts_every_call(self):
        from .test_medqa_bounded_path import NewBooks
        class Followup(Model):
            def invoke(self, messages):
                response = super().invoke(messages)
                data = json.loads(response.content)
                payload = json.loads(messages[1].content)
                if 'final_answer' in data and not payload.get('feedback', {}).get('followup_completed'):
                    data['followup_request'] = {'gap':'synthetic decisive missing relation', 'basis':'could change choice',
                        'expected_information':'a distinct relation', 'may_change_answer':True,
                        'knowledge_task_ids':['K1'], 'evidence_ids':[], 'queries':['new distinct fixture query']}
                response.content = json.dumps(data)
                return response
        model = Followup()
        agents = ChatRoles(model, output_policy='decoupled', memory_selection='needs', workspace_policy=POLICY)
        result = Runtime(agents, NewBooks(), output_policy='decoupled', memory_policy='basic', memory_selection='needs',
                         workspace_policy=POLICY, max_calls=4).run(self.q)
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(result['metrics']['model_calls'], 4)
        self.assertEqual(result['metrics']['followup_rounds'], 1)
        self.assertEqual(result['metrics']['total_tokens'], 480)
        self.assertIn('Distinct synthetic excerpt', model.messages[-1][1].content)

    def test_vector_still_reaches_real_receiver_without_another_model_call(self):
        class Memory:
            def __init__(self): self.received = []
            def recall(self, values, *args, **kwargs): self.received.append(values); return []
        memory = Memory()
        result, _, _ = self.run_case(memory=memory, embedder=Embeddings(), memory_write=False)
        self.assertEqual(len(memory.received), 1)
        self.assertEqual(list(memory.received[0]), [1., 0., 0., 0.])
        self.assertEqual(result['metrics']['state_handle_reads'], 1)
        self.assertEqual(result['metrics']['model_calls'], 3)

    def test_ten_tasks_with_persistent_memory_are_stable_and_can_reuse_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            with closing(SharedMemory(Path(folder)/'memory.sqlite3', 'workspace', 'fixture')) as memory:
                rows = []
                for index in range(10):
                    self.q = Question('q'+str(index), self.q.question, self.q.options)
                    result, _, _ = self.run_case(memory=memory, embedder=Embeddings())
                    self.assertEqual(result['answer'], 'A', result)
                    self.assertEqual(result['metrics']['model_calls'], 3)
                    self.assertEqual(result['metrics']['memory_write_errors'], 0)
                    rows.append(result)
                self.assertTrue(any(r['reused_memory_ids'] for r in rows[1:]))

    def test_crash_replay_uses_saved_projected_requests_without_rebilling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); benchmark = root/'benchmark.json'; out = root/'run'
            benchmark.write_text(json.dumps({'medqa': {'q': {**self.q.public(), 'answer':'A'}}}))
            books = Books(); books.warmup = books.close = lambda: None
            embedding = SimpleNamespace(_model=object(), cache_key='fixture', cache_store=None)
            argv = ['medqa', 'run', '--benchmark', str(benchmark), '--output', str(out), '--split', 'all', '--limit', '1',
                    '--memory','off','--output-policy','decoupled','--memory-policy','basic','--memory-selection','needs',
                    '--workspace-policy', POLICY]
            first = Model(); second = Model()
            original_execute = DurableTools.execute
            def cut_after_search(journal, kind, request, callback):
                value = original_execute(journal, kind, request, callback)
                if kind == 'search_textbooks': raise PowerCut()
                return value
            def run(model, resume=False):
                agents = ChatRoles(model, output_policy='decoupled', memory_selection='needs', workspace_policy=POLICY)
                with (patch('sys.argv', argv + (['--resume'] if resume else [])),
                      patch.object(cli, 'books', return_value=books),
                      patch.object(cli, 'create_model', return_value=(agents, {'model':'fixture'})),
                      patch('LangChain.memory.embedding.QwenEmbeddingService', return_value=embedding), patch('builtins.print')):
                    cli.main()
            with patch.object(DurableTools, 'execute', cut_after_search), self.assertRaises(PowerCut):
                run(first)
            run(second, resume=True)
            self.assertEqual(len(first.messages), 2)
            self.assertEqual(len(second.messages), 1)
            result = json.loads((out/'structured_off.jsonl').read_text())
            self.assertEqual(result['answer'], 'A')
            self.assertEqual(result['metrics']['checkpoint_replayed_calls'], 2)
            self.assertEqual(result['metrics']['total_tokens'], 360)


if __name__ == '__main__':
    unittest.main()
