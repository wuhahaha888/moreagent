"""Follow-up delivery regressions; synthetic evidence, no paid/medical claims."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .test_medqa import Books
from .test_medqa_bounded_path import PlanAgents, query
from .test_medqa_checkpoint import PowerCut
from .test_medqa_v6 import document
from ..agents.medqa_prompts import REASONER
from ..medqa.checkpoint import Checkpoint, DurableCalls, DurableTools
from ..medqa.data import Question
from ..medqa.evidence_processor import EvidenceProcessor, validate_view
from ..medqa.evidence_handoff import build_handoff, visible_delta
from ..medqa.retrieval import digest
from ..medqa.runtime import Runtime


def citation(doc, quote=None, support='supported'):
    return {'evidence_id': doc['id'], 'quote': quote or doc['text'], 'source_verified': True,
            'visible': True, 'model_reported_support': support}


REQUEST = {'gap': 'decisive missing relation', 'expected_information': 'novel restriction',
           'knowledge_task_ids': ['K1'], 'evidence_ids': [], 'queries': ['decisive missing relation']}


class BulkBooks(Books):
    def search(self, text):
        self.calls += 1
        self.stats['searches'] += 1
        if self.calls <= 2:
            return [document(f'Old{self.calls}_{i}',
                    f'Initial synthetic condition holds for eligible adults. Background source {self.calls} number {i}.')
                    for i in range(4)]
        return [document(f'New_{i}',
                f'Novel counterexample for the decisive missing relation. This restriction excludes children. Source {i}.')
                for i in range(4)]


class HandoffAgents(PlanAgents):
    def __init__(self, queries=None, gap=None):
        super().__init__(queries=queries, followup=True)
        self.gap = gap

    def call(self, role, payload, mode):
        original, usage = super().call(role, payload, mode)
        if role != REASONER:
            return original, usage
        pool = payload['evidence_pool']
        updated = payload.get('feedback', {}).get('followup_completed')
        answer = 'B' if updated and any('counterexample' in d['text'] for d in pool) else 'A'
        data = {'question_id': payload['question_id'], 'decision': 'answer',
                'final_answer': {'answer': answer, 'evidence_status': 'partial', 'summary': 'synthetic result'},
                'citations': [{'evidence_id': pool[0]['id'], 'quote': pool[0]['text'].split('\n[未展开区间]\n')[0][:200],
                               'support': 'supported'}] if pool else [],
                'critical_gaps': ['synthetic missing relation'],
                'followup_request': deepcopy(original['followup_request'])}
        if self.gap:
            data['followup_request'].update(gap=self.gap, expected_information=self.gap, queries=[self.gap])
        return data, usage


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.q = Question('q', 'Initial synthetic condition with NOT and a time limit?', {k:k for k in 'ABCD'})

    def run_case(self, agents, books, **kwargs):
        trace = kwargs.pop('trace_id', None)
        kwargs.setdefault('max_calls', 4)
        kwargs.setdefault('max_searches', 3)
        return Runtime(agents, books, output_policy='decoupled', **kwargs).run(self.q, trace_id=trace)

    def bulk_agents(self):
        return HandoffAgents([query('initial first'), query('initial complementary', 'Q2')])

    def test_full_old_pool_delivers_new_body_in_both_modes_with_processor_on_or_off(self):
        for enabled in (True, False):
            payloads = []
            for mode in ('text', 'structured'):
                with self.subTest(enabled=enabled, mode=mode):
                    agents, books = self.bulk_agents(), BulkBooks()
                    result = self.run_case(agents, books, evidence_processor=enabled, mode=mode)
                    first, last = [p for role, p, _ in agents.calls if role == REASONER]
                    self.assertEqual(len(first['evidence_pool']), 8)
                    self.assertTrue(any(d['id'].startswith('New_') for d in last['evidence_pool']))
                    self.assertTrue(any('counterexample' in d['text'] for d in last['evidence_pool']))
                    self.assertLess(len(last['evidence_pool']), 8)  # Do not refill with already-seen background.
                    self.assertIn(first['evidence_pool'][0]['text'], '\n'.join(d['text'] for d in last['evidence_pool']))
                    self.assertLessEqual(len(last['evidence_pool']), 8)
                    self.assertLessEqual(sum(len(d['text']) for d in last['evidence_pool']), 12000)
                    self.assertEqual((result['answer'], result['metrics']['model_calls']), ('B', 4))
                    self.assertEqual(result['metrics']['retrieval_count'], 3)
                    self.assertGreater(result['metrics']['followup_new_visible_chars'], 0)
                    self.assertEqual(result['metrics']['evidence_raw_chars'],
                                     result['evidence_processing']['raw_chars'])
                    self.assertEqual(len(result['evidence_processing']['candidate_coverage']), 12)
                    self.assertEqual(last['question'], self.q.question)
                    self.assertEqual(last['options'], self.q.options)
                    payloads.append(last['evidence_pool'])
            self.assertEqual(payloads[0], payloads[1])

    def test_allocation_adapts_to_one_or_many_new_sources_without_half_quota(self):
        old = {f'O{i}': document(f'O{i}', f'Old condition evidence for initial context number {i}.') for i in range(8)}
        processor = EvidenceProcessor()
        previous = processor.process(old, query='initial context')
        numbers = []
        for count in (1, 10):
            docs = {**old, **{f'N{i}': document(f'N{i}', f'Decisive missing relation and novel restriction number {i}.') for i in range(count)}}
            pack = build_handoff(processor, docs, previous=previous,
                citations=[citation(d) for d in old.values()], request=REQUEST, question='initial context')
            self.assertIn('O0', pack['visible'])
            numbers.append(sum(v['id'].startswith('N') for v in pack['views']))
            validate_view(pack, docs)
        self.assertEqual(numbers[0], 1)
        self.assertGreater(numbers[1], 4)

    def test_distinct_old_needs_keep_their_anchors_before_redundant_gap_sources(self):
        old = {f'O{i}':document(f'O{i}',f'Old condition with distinct evidence number {i}.') for i in range(8)}
        new = {f'N{i}':document(f'N{i}',f'Decisive missing relation novel restriction number {i}.') for i in range(10)}
        targets = {f'O{i}':['K'+str(min(i+1,3))] for i in range(8)}
        targets.update({i:['K4'] for i in new})
        previous = EvidenceProcessor().process(old, query='condition')
        pack = build_handoff(EvidenceProcessor(),{**old,**new},previous=previous,
            citations=[citation(old['O0']),citation(old['O1']),citation(old['O2'])],
            targets=targets,request={**REQUEST,'knowledge_task_ids':['K4']},question='condition')
        self.assertTrue({'O0','O1','O2'} <= set(pack['visible']))
        self.assertTrue(any(i.startswith('N') for i in pack['visible']))

    def test_seen_best_window_can_yield_to_a_relevant_unseen_window_from_same_source(self):
        prefix = 'Decisive missing relation novel restriction. '*25
        text = prefix+'Background paragraph. '*100+'Decisive relation has a newly stated exception. '
        doc = document('same',text)
        processor = EvidenceProcessor(span_chars=180)
        previous = {'views':[{'id':'same','text':prefix}],
                    'visible':{'same':[{'start':0,'end':len(prefix),'text':prefix}]}}
        pack = build_handoff(processor,{'same':doc},previous=previous,citations=[],request=REQUEST,question='initial')
        self.assertGreater(pack['handoff']['new_visible_chars'],0)
        self.assertNotEqual(pack['views'],previous['views'])

    def test_reported_counterevidence_and_negation_are_retained(self):
        positive = document('old', 'A synthetic rule is recommended for adults only. Exclude children.')
        negative = document('new', 'A synthetic rule is not recommended for adults only. Exclude children.')
        docs = {'old': positive, 'new': negative}
        previous = EvidenceProcessor().process({'old': positive}, query='rule')
        pack = build_handoff(EvidenceProcessor(max_documents=2), docs, previous=previous,
            citations=[citation(positive)], request=REQUEST, question='rule',
            conflicts=[{'evidence_ids':['old', 'new'], 'status':'unresolved'}])
        self.assertEqual({v['id'] for v in pack['views']}, {'old', 'new'})
        self.assertIn('not recommended', pack['visible']['new'][0]['text'])
        self.assertEqual(pack['handoff']['conflict_ids'], ['new', 'old'])

    def test_same_source_unseen_gap_span_counts_as_new_information(self):
        text = ('Initial synthetic condition has an anchor. ' * 80
                + 'A novel gapmarker counterexample has a restriction. Except in children. '
                + 'Unrelated background. ' * 80)
        doc = document('same', text)
        class SameSource(Books):
            def search(self, query_text):
                self.calls += 1
                self.stats['searches'] += 1
                return [deepcopy(doc)]
        agents = HandoffAgents(gap='novel gapmarker restriction')
        result = self.run_case(agents, SameSource(), evidence_span_chars=180, max_evidence_chars=2400)
        self.assertEqual(result['metrics']['model_calls'], 4)
        self.assertEqual(result['answer'], 'B')
        self.assertEqual(result['metrics']['followup_new_visible_sources'], 1)
        last = agents.calls[-1][1]
        self.assertIn('gapmarker', last['evidence_pool'][0]['text'])
        self.assertIn('Except in children.', last['evidence_pool'][0]['text'])
        self.assertEqual(last['feedback']['new_visible_evidence_ids'], ['same'])

    def test_alias_of_already_seen_source_is_not_progress(self):
        class AliasBooks(Books):
            def search(self, text):
                docs = super().search(text)
                return [{**d, 'id':'Alias_'+str(self.calls)} for d in docs]
        result = self.run_case(HandoffAgents(), AliasBooks())
        self.assertEqual(result['metrics']['model_calls'], 3)
        self.assertEqual(result['metrics']['followup_new_visible_chars'], 0)
        self.assertEqual(result['followup_result']['reason'], 'followup_no_new_visible_evidence')

    def test_invalid_old_citation_cannot_block_a_valid_followup_answer(self):
        class BadCitation(HandoffAgents):
            def call(self, role, payload, mode):
                value, usage = super().call(role,payload,mode)
                if role==REASONER and not payload.get('feedback',{}).get('followup_completed'):
                    value['citations'][0]['quote']='Invented old quote which does not occur in the source.'
                return value,usage
        agents = BadCitation([query('initial first'),query('initial complementary','Q2')])
        result = self.run_case(agents,BulkBooks())
        self.assertEqual((result['answer'],result['metrics']['model_calls']),('B',4))
        self.assertFalse(result['protocol_result']['blocks_answer'])
        self.assertEqual(result['evidence_processing']['handoff']['omitted_anchor_ids'],[])

    def test_same_excerpt_in_different_raw_sources_is_not_new_visible_information(self):
        text = 'Identical visible excerpt with important conditions. ' * 40
        old = document('old', text+'Old hidden footer.')
        new = document('new', text+'Different hidden footer.')
        previous = {'views':[{'id':'old', 'text':text[:500]}],
                    'visible':{'old':[{'start':0, 'end':500, 'text':text[:500]}]}}
        shown = {'views':[{'id':'new', 'text':text[:500]}],
                 'visible':{'new':[{'start':0, 'end':500, 'text':text[:500]}]}}
        self.assertEqual(visible_delta(shown, previous, {'old':old,'new':new})['characters'], 0)

    def test_expanding_anchor_margins_cannot_fake_new_gap_information(self):
        doc = document('old', 'Background. '*100+'Initial anchor must retain its visible conditions. '+'Background. '*100)
        a = doc['text'].index('Initial anchor')
        text = doc['text'][a:a+len('Initial anchor must retain its visible conditions.')]
        previous = {'views':[{'id':'old', 'text':text}],
                    'visible':{'old':[{'start':a,'end':a+len(text),'text':text}]}}
        # No gap keywords appear outside this already-visible interval.
        req = {**REQUEST, 'gap':'Initial anchor', 'expected_information':'Initial anchor', 'queries':['Initial anchor']}
        pack = build_handoff(EvidenceProcessor(span_chars=20), {'old':doc}, previous=previous,
            citations=[citation(doc,text)], request=req, question='Initial anchor')
        # The gap window itself may legitimately add adjacent source context,
        # but the anchor range must remain bounded by the actually seen interval.
        anchors = [r for r in pack['records'] if r['action']=='handoff_anchor']
        self.assertEqual(anchors[0]['ranges'], [(a,a+len(text))])

    def test_too_large_new_full_document_preserves_answer_without_update_call(self):
        class HugeBooks(Books):
            def search(self, text):
                docs = super().search(text)
                return docs if self.calls==1 else [document('huge', 'A novel counterexample. '*2000)]
        agents = HandoffAgents()
        result = self.run_case(agents, HugeBooks(), evidence_processor=False, max_evidence_chars=1000)
        self.assertEqual((result['answer'],result['metrics']['model_calls']), ('A',3))
        self.assertEqual(result['followup_result']['reason'], 'followup_no_new_visible_evidence')

    def test_one_document_limit_reports_omitted_old_anchor_and_still_delivers_new_body(self):
        old, new = document('old','Initial support with necessary context.'), document('new','Decisive missing relation is a counterexample.')
        previous = EvidenceProcessor().process({'old':old}, query='initial')
        pack = build_handoff(EvidenceProcessor(max_documents=1), {'old':old,'new':new},
                             previous=previous,citations=[citation(old)],request=REQUEST,question='initial')
        self.assertEqual([v['id'] for v in pack['views']], ['new'])
        self.assertEqual(pack['handoff']['omitted_anchor_ids'], ['old'])
        self.assertGreater(pack['handoff']['new_visible_chars'], 0)

    def test_raw_new_sources_without_new_final_view_do_not_trigger_an_update(self):
        def stale(processor, documents, **kw):
            pack = deepcopy(kw['previous'])
            pack['handoff'] = {'new_visible_ids':['invented'], 'new_visible_chars':999,
                               'omitted_anchor_ids':[], 'visible_ids':[v['id'] for v in pack['views']]}
            return pack
        with patch('LangChain.medqa.runtime.build_handoff', stale):
            result = self.run_case(self.bulk_agents(), BulkBooks())
        self.assertEqual((result['answer'],result['metrics']['model_calls']), ('A',3))
        self.assertEqual(result['metrics']['followup_new_visible_chars'],0)

    def test_durable_handoff_replay_does_not_repeat_committed_searches_or_agent_calls(self):
        with tempfile.TemporaryDirectory() as root:
            with Checkpoint(Path(root)) as cp:
                original = DurableTools.execute
                def cut_after_handoff(journal, kind, request, callback):
                    value = original(journal,kind,request,callback)
                    if kind=='evidence_handoff':
                        raise PowerCut()
                    return value
                with patch.object(DurableTools,'execute',cut_after_handoff):
                    with self.assertRaises(PowerCut):
                        self.run_case(DurableCalls(self.bulk_agents(),cp,'structured_off',0),BulkBooks(),
                                      journal=DurableTools(cp,'structured_off',0),trace_id='fixed')
                agents, books = self.bulk_agents(), BulkBooks()
                result = self.run_case(DurableCalls(agents,cp,'structured_off',0),books,
                                      journal=DurableTools(cp,'structured_off',0),trace_id='fixed')
                self.assertEqual(books.calls,0)
                self.assertEqual(len(agents.calls),1)
                self.assertEqual(result['answer'],'B')
                self.assertEqual(result['metrics']['total_tokens'],60)


if __name__ == '__main__':
    unittest.main()
