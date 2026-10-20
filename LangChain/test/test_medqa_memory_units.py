"""Basic shared-memory lifecycle with real SQLite/IPC and synthetic Agents."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .test_medqa import Books, Embeddings
from .test_medqa_bounded_path import PlanAgents
from .test_medqa_checkpoint import PowerCut
from ..agents.medqa_prompts import PLANNER,INVESTIGATOR,REASONER
from ..medqa.checkpoint import Checkpoint,PendingMemory
from ..medqa.data import Question
from ..medqa.memory import SharedMemory
from ..medqa.memory_units import normalize_unit
from ..medqa.runtime import Runtime
from ..medqa import cli


class MinimalAgents(PlanAgents):
    def __init__(self, *, reuse=False, invalid_quote=False, no_answer=False):
        super().__init__()
        self.reuse,self.invalid_quote,self.no_answer=reuse,invalid_quote,no_answer

    def call(self,role,payload,mode):
        data,usage=super().call(role,payload,mode)
        if role==INVESTIGATOR and self.reuse and payload['evidence_pool']:
            data={'question_id':payload['question_id'],'action':'finish'}
        if role==REASONER:
            doc=payload['evidence_pool'][0] if payload['evidence_pool'] else None
            data={'question_id':payload['question_id'],'decision':'answer',
                  'final_answer':{'answer':None if self.no_answer else 'A','evidence_status':'partial','summary':'synthetic partial answer'},
                  'citations':[{'evidence_id':doc['id'],'quote':'Invented citation not in source.' if self.invalid_quote else doc['text'],
                                'support':'partial'}] if doc else [],'critical_gaps':['synthetic unresolved scope']}
        return data,usage


class MemoryUnitsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.memory=SharedMemory(self.root/'memory.sqlite3','typed','fixture-model')
        self.addCleanup(self.memory.close)
        self.q=Question('first','Synthetic case with a NOT and a time condition?',{k:k for k in 'ABCD'})

    def run_case(self, question=None, agents=None, memory=None, **kw):
        kw.setdefault('memory_policy','basic')
        return Runtime(agents or MinimalAgents(),Books(),output_policy='decoupled',max_calls=4,
                       memory=self.memory if memory is None else memory,embedder=Embeddings(),task_group='g',**kw).run(question or self.q)

    def records(self):
        with sqlite3.connect(self.root/'memory.sqlite3') as db:
            db.row_factory=sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM memory ORDER BY kind')]

    def test_partial_answer_writes_three_units_with_required_metadata(self):
        result=self.run_case()
        self.assertFalse(result['evidence_result']['memory_eligible'])
        self.assertEqual(result['metrics']['memory_written_by_kind'],{'evidence':1,'strategy':1,'episode':1})
        self.assertEqual(result['metrics']['model_calls'],3)
        rows=self.records()
        self.assertEqual({r['kind'] for r in rows},{'evidence','strategy','episode'})
        for row in rows:
            for key in ['id','source_agent','created_at','task_topic','summary']:
                self.assertTrue(row[key])
            data=json.loads(row['content'])
            self.assertNotIn('answer',data)
            self.assertNotIn('correct',data)
            self.assertNotIn('text',data)
            self.assertNotIn('document',data)
            self.assertEqual(data['task_group'],'g')
            self.assertIn('validation',data)

    def test_later_agents_reuse_typed_memories_without_using_experience_as_evidence(self):
        self.run_case()
        agents=MinimalAgents(reuse=True)
        result=self.run_case(Question('second',self.q.question,self.q.options),agents)
        planner=next(p for role,p,_ in agents.calls if role==PLANNER)
        investigator=next(p for role,p,_ in agents.calls if role==INVESTIGATOR)
        reasoner=next(p for role,p,_ in agents.calls if role==REASONER)
        self.assertEqual({h['kind'] for h in planner['memory_hits']},{'evidence','strategy','episode'})
        strategy=[h for h in investigator['memory_hits'] if h.get('kind')=='strategy']
        self.assertTrue(strategy and strategy[0]['retrieval_queries'])
        episode=next(h for h in investigator['memory_hits'] if h.get('kind')=='episode')
        self.assertTrue(episode['episode_context']['needs'])
        self.assertIn('synthetic unresolved scope',episode['episode_context']['gaps'])
        planner_episode=next(h for h in planner['memory_hits'] if h.get('kind')=='episode')
        self.assertIn('synthetic unresolved scope',planner_episode['summary'])
        self.assertFalse(any(h.get('kind') in {'strategy','episode'} for h in reasoner['memory_hits']))
        self.assertTrue(result['reused_memory_ids'])
        self.assertEqual(result['metrics']['retrieval_count'],0)
        self.assertEqual(result['metrics']['document_reads'],1)
        self.assertEqual(result['metrics']['memory_experience_reads'],2)
        self.assertEqual(cli.summarize([{**result,'correct':False}])['memory_source_check_coverage'],1)

    def test_invalid_quote_does_not_enter_evidence_memory(self):
        result=self.run_case(agents=MinimalAgents(invalid_quote=True))
        self.assertEqual(result['answer'],'A')
        self.assertEqual(result['metrics']['memory_written_by_kind'],{'evidence':0,'strategy':1,'episode':1})
        self.assertEqual({r['kind'] for r in self.records()},{'strategy','episode'})

    def test_planner_rejection_prevents_experience_delivery(self):
        self.run_case()
        class RejectExperience(MinimalAgents):
            def call(self,role,payload,mode):
                data,usage=super().call(role,payload,mode)
                if role==PLANNER:
                    kinds={h['id']:h['kind'] for h in payload['memory_hits']}
                    for assessment in data['memory_assessments']:
                        if kinds[assessment['memory_id']] in {'strategy','episode'}:
                            assessment.update(decision='reject',reason='unrelated execution context',knowledge_task_ids=[])
                return data,usage
        agents=RejectExperience()
        result=self.run_case(Question('later',self.q.question,self.q.options),agents)
        self.assertEqual(result['metrics']['memory_experience_reads'],0)
        investigator=next(p for role,p,_ in agents.calls if role==INVESTIGATOR)
        self.assertFalse(any(h.get('kind') in {'strategy','episode'} for h in investigator['memory_hits']))
        self.assertEqual(result['answer'],'A')

    def test_missing_ids_do_not_promote_summaries_to_evidence_or_block_search(self):
        self.run_case()
        agents=MinimalAgents()
        with patch.object(self.memory,'resolve',return_value=[]):
            result=self.run_case(Question('later',self.q.question,self.q.options),agents)
        self.assertEqual(result['metrics']['memory_experience_reads'],0)
        self.assertEqual(result['metrics']['memory_source_valid'],0)
        self.assertEqual(result['metrics']['retrieval_count'],1)
        self.assertEqual(result['answer'],'A')
        investigator=next(p for role,p,_ in agents.calls if role==INVESTIGATOR)
        self.assertEqual(investigator['memory_hits'],[])

    def test_basic_prefetch_encodes_all_options_without_grading_answer(self):
        class RecordingEmbedding(Embeddings):
            def __init__(self):self.texts=[]
            def embed_text(self,text):
                self.texts.append(text)
                return super().embed_text(text)
        embedding=RecordingEmbedding()
        question=Question('short','Saliva is increased by:',{'A':'Cholinergic drugs','B':'Anticholinergic drugs'})
        Runtime(MinimalAgents(),Books(),output_policy='decoupled',memory_policy='basic',max_calls=4,
                memory=self.memory,embedder=embedding,task_group='g').run(question)
        self.assertEqual(embedding.texts[0],'g\nSaliva is increased by:\nOptions:\nA: Cholinergic drugs\nB: Anticholinergic drugs')

    def test_failed_answer_can_save_execution_observations(self):
        result=self.run_case(agents=MinimalAgents(no_answer=True,invalid_quote=True))
        self.assertIsNone(result['answer'])
        self.assertTrue(any(r['kind']=='episode' for r in self.records()))
        self.assertFalse(any(r['kind']=='evidence' for r in self.records()))

    def test_unknown_usage_does_not_publish_reusable_units(self):
        class Unknown(MinimalAgents):
            def call(self,role,payload,mode):
                value,usage=super().call(role,payload,mode)
                return value,({'source':'unknown'} if role==REASONER else usage)
        result=self.run_case(agents=Unknown())
        self.assertEqual(result['answer'],'A')
        self.assertEqual(self.records(),[])

    def test_default_strict_and_read_only_keep_their_previous_meaning(self):
        for settings in ({'memory_policy':'strict'},{'memory_write':False}):
            result=self.run_case(**settings)
            self.assertEqual(result['metrics']['memory_written'],0)
        self.assertEqual(self.records(),[])

    def test_resolve_scope_and_type_filter_apply_to_all_units(self):
        result=self.run_case()
        rows=self.records()
        ids=[r['id'] for r in rows]
        self.assertEqual(self.memory.resolve(ids,result['trace_id'],Books.version,task_group='g'),[])
        self.assertEqual(self.memory.resolve(ids,'later','wrong',task_group='g'),[])
        self.assertEqual(self.memory.resolve(ids,'later',Books.version,task_group='wrong'),[])
        resolved=self.memory.resolve(ids,'later',Books.version,task_group='g')
        self.assertEqual(len(resolved),3)
        hits=self.memory.recall([1.,0.,0.,0.],'later',Books.version,task_group='g',kinds=['strategy'])
        self.assertEqual([h['kind'] for h in hits],['strategy'])
        self.assertNotIn('queries',hits[0]['content'])  # detail is fetched by ID
        self.assertTrue(next(r for r in resolved if r['kind']=='strategy')['content']['queries'])

    def test_generic_units_and_outbox_retry_are_idempotent_after_commit_cut(self):
        staged=PendingMemory(self.memory)
        result=self.run_case(memory=staged)
        self.assertEqual(len(staged.writes),3)
        with Checkpoint(self.root) as cp:
            cp.begin('structured_on',0,self.q.id)
            cp.save_result('structured_on',0,result,staged.writes)
            original=self.memory.put_unit
            def cut(unit,values,trace):
                original(unit,values,trace)
                raise PowerCut()
            with patch.object(self.memory,'put_unit',cut):
                with self.assertRaises(PowerCut):cp.apply_memory('structured_on',self.memory)
            cp.apply_memory('structured_on',self.memory)
        self.assertEqual(len(self.records()),3)

    def test_forged_source_and_grading_fields_are_rejected(self):
        staged=PendingMemory(self.memory)
        self.run_case(memory=staged)
        evidence=deepcopy(next(w['unit'] for w in staged.writes if w['unit']['kind']=='evidence'))
        evidence['content']['excerpts']=['Invented source statement.']
        with self.assertRaises(ValueError):normalize_unit(evidence,'trace')
        episode=deepcopy(next(w['unit'] for w in staged.writes if w['unit']['kind']=='episode'))
        episode['content']['gold_answer']='A'
        with self.assertRaises(ValueError):normalize_unit(episode,'trace')

    def test_ten_consecutive_tasks_persist_and_reuse_shared_memory(self):
        results=[]
        for n in range(10):
            results.append(self.run_case(Question('round-'+str(n),self.q.question,self.q.options),MinimalAgents(reuse=True)))
        self.assertTrue(all(r['answer']=='A' and r['metrics']['model_calls']==3 for r in results))
        self.assertTrue(all(r['metrics']['memory_used']>0 for r in results[1:]))
        self.assertEqual(sum(r['metrics']['retrieval_count'] for r in results),1)
        self.assertEqual(sum(r['kind']=='episode' for r in self.records()),10)


class BasicMemoryEntryTest(unittest.TestCase):
    def test_four_online_conditions_and_resume_preserve_completed_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            benchmark=root/'benchmark.json'
            benchmark.write_text(json.dumps({'medqa':{key:{'question':'synthetic related case',
                'options':{k:k for k in 'ABCD'},'answer':'A'} for key in ['one','two']}}))
            output=root/'comparison'
            agents=MinimalAgents(reuse=True)
            def run(resume=False):
                books=Books()
                books.warmup=books.close=lambda:None
                embedding=SimpleNamespace(_model=object(),cache_key='fixture-model',cache_store=None,
                    embed_text=Embeddings().embed_text)
                argv=['medqa','compare','--benchmark',str(benchmark),'--split','all','--all',
                    '--output',str(output),'--output-policy','decoupled','--memory-policy','basic','--memory-selection','needs',
                    '--memory-write','on','--max-calls','4','--max-total-tokens','10000']
                if resume:argv.append('--resume')
                with patch('sys.argv',argv),patch.object(cli,'books',return_value=books),\
                     patch.object(cli,'create_model',return_value=(agents,{'model':'fixture'})),\
                     patch('LangChain.memory.embedding.QwenEmbeddingService',return_value=embedding),\
                     patch('builtins.print'):
                    cli.main()
            run()
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(set(report),{'text_off','structured_off','text_on','structured_on'})
            counts={}
            for condition,stats in report.items():
                self.assertEqual(stats['questions_completed'],2)
                self.assertFalse(stats['token_usage_complete'])  # synthetic usage is not provider usage
                if condition.endswith('_on'):
                    self.assertTrue(all(stats['memory_written_by_kind'][k]>0 for k in ['evidence','strategy','episode']))
                    self.assertGreater(stats['memory_experience_reads'],0)
                    with sqlite3.connect(output/(condition+'.sqlite3')) as db:
                        counts[condition]=db.execute('SELECT count(*) FROM memory').fetchone()[0]
                else:self.assertEqual(stats['memory_written'],0)
            calls=len(agents.calls)
            run(resume=True)
            self.assertEqual(len(agents.calls),calls)
            for condition,count in counts.items():
                with sqlite3.connect(output/(condition+'.sqlite3')) as db:
                    self.assertEqual(db.execute('SELECT count(*) FROM memory').fetchone()[0],count)


if __name__=='__main__':unittest.main()
