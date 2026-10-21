"""Need-first execution tests; fixtures prove control flow, not API savings."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .test_medqa import Books, Embeddings
from .test_medqa_checkpoint import PowerCut
from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from ..medqa import cli
from ..medqa.checkpoint import DurableTools, Checkpoint
from ..medqa.data import Question
from ..medqa.grounding import ContractError
from ..medqa.memory import SharedMemory
from ..medqa.memory_units import embedding_text
from ..medqa.need_flow import investigation_reviews, reasoner_reviews, select_sources, compact_memory_view, fragment_citations
from ..medqa.retrieval import digest
from ..medqa.runtime import Runtime, ChatRoles
from ..medqa.task_state import TaskStates
from ..medqa.workspace import POLICY


class NeedAgents:
    def __init__(self, *, two=False, reject=False, follow=False, overquery=False, invalid=False):
        self.calls=[]; self.two=two; self.reject=reject; self.follow=follow
        self.overquery=overquery; self.invalid=invalid; self.reasoner_calls=0

    def call(self, role, p, mode):
        self.calls.append((role,deepcopy(p),mode))
        out={'question_id':p['question_id']}
        if role==PLANNER:
            out.update(key_conditions=[{'id':'C1','text':'Synthetic adult NOT condition.'}],
                knowledge_tasks=[{'id':ident,'goal':'verify synthetic relation '+ident,
                    'condition_ids':['C1'],'depends_on':[]} for ident in (['K1','K2'] if self.two else ['K1'])],
                criteria=['source supported relation'],memory_assessments=[])
        elif role==INVESTIGATOR:
            docs=p.get('evidence_pool',[]); covered=[]
            for task in p['knowledge_tasks']:
                ident=task['id']; usable=bool(docs) and not self.reject and ident=='K1' and not p.get('feedback',{}).get('reopen_need_ids')
                if usable:covered.append(ident)
            out['need_updates']=[{'need_id':t['id'],
                'coverage':'covered' if t['id'] in covered else 'unusable' if docs and self.reject else 'missing',
                'evidence_ids':[docs[0]['id']] if t['id'] in covered else [],
                'condition_verdict':'matched' if t['id'] in covered else 'unknown',
                'reason':'synthetic source relation matches' if t['id'] in covered else 'decisive relation missing'}
                for t in p['knowledge_tasks']]
            active=p.get('active_need_ids') or [t['id'] for t in p['knowledge_tasks']]
            requests=[i for i in active if i not in covered or self.overquery]
            out['action']='tool_plan' if requests else 'finish'
            if requests:
                out['search']={'gap':'missing synthetic relation','queries':[
                    {'id':'Q'+ident,'query':'new distinct fixture query' if self.follow and p.get('feedback',{}).get('reopen_need_ids')
                         else 'fixture query '+ident,'knowledge_task_ids':[ident],
                     'expected_information':'decisive synthetic relation','run_if':'always'} for ident in requests]}
            if self.invalid:out['need_updates'][0]['coverage']='invented'
        else:
            self.reasoner_calls+=1
            docs=p.get('evidence_pool',[])
            first_follow=self.follow and self.reasoner_calls==1
            out.update(decision='answer',final_answer={'answer':'B' if self.follow and not first_follow else 'A',
                'evidence_status':'partial' if first_follow else 'supported','summary':'synthetic only'},
                citations=[{'evidence_id':d['id'],'quote':d['text'],'support':'supported'} for d in docs],
                critical_gaps=['decisive relation missing'] if first_follow else [],
                need_reviews=[{'task_id':t['id'],'verdict':'partial' if first_follow and docs else 'supported' if docs else 'insufficient',
                    'citation_indices':[0] if docs else [], 'condition_verdict':'matched' if docs else 'unknown',
                    'reason':'synthetic relation review'} for t in p['knowledge_tasks']],
                next_action='investigate' if first_follow else 'finish')
            if first_follow:
                out['investigation_request']={'need_ids':['K1'],'gap':'decisive relation missing',
                    'expected_information':'distinct synthetic relation','may_change_answer':True}
        return out, {'prompt_tokens':10,'completion_tokens':5,'total_tokens':15,
                     'source':'fixture','response_model':'fixture'}


class NewBooks(Books):
    def search(self,query):
        docs=super().search(query)
        if 'new distinct' in query:
            doc=deepcopy(self.doc);doc.update(id='Fixture_1',text='Distinct synthetic excerpt supplies another relation.')
            doc['sha256']=digest(doc['text']);return [doc]
        return docs


class NeedChat:
    """Exercise the real ChatRoles prompt and serializer boundary locally."""
    def __init__(self):self.messages=[];self.agents=NeedAgents()
    def invoke(self,messages):
        self.messages.append(messages)
        prompt=messages[0].content
        role=PLANNER if '你是Planner' in prompt else INVESTIGATOR if '你是Investigator' in prompt else REASONER
        out,_=self.agents.call(role,json.loads(messages[1].content),'structured')
        return SimpleNamespace(content=json.dumps(out),response_metadata={'model':'local-fixture',
            'finish_reason':'stop','token_usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})


class NeedFirstTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.q=Question('first','Synthetic adult NOT condition. Which relation applies?',{k:k for k in 'ABCD'})

    def memory(self):
        memory=SharedMemory(self.root/'memory.sqlite3','need-first','fixture')
        self.addCleanup(memory.close);return memory

    def run_case(self,agents=None,memory=None,books=None,**kwargs):
        kwargs.setdefault('max_calls',7)
        runtime=Runtime(agents or NeedAgents(),books or Books(),memory=memory,
            embedder=Embeddings() if memory else None, output_policy='decoupled',memory_policy='basic',
            memory_selection='needs',workspace_policy=POLICY,workflow='need_first',task_group='g',**kwargs)
        return runtime.run(self.q)

    def seed(self,memory):
        result=self.run_case(memory=memory)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['memory_written_by_kind'],{'evidence':1,'strategy':0,'episode':0},result)
        self.q=Question('later',self.q.question,self.q.options)
        return result

    def test_planner_runs_first_and_need_recall_omits_entire_case(self):
        order=[]
        class Tracked(NeedAgents):
            def call(self,role,p,mode):order.append(role);return super().call(role,p,mode)
        class Memory:
            def recall(self,*args,**kw):order.append('recall');return []
        result=self.run_case(Tracked(),memory=Memory(),memory_write=False)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(order[:3],[PLANNER,'recall',INVESTIGATOR])
        event=next(e for e in result['events'] if e.get('tool')=='memory_prefetch')
        self.assertEqual(event['before_agent'],INVESTIGATOR)
        self.assertEqual(event['need_id'],'K1')
        self.assertFalse(next(e for e in result['events'] if e.get('agent')==PLANNER)['input'].get('memory_hits'))
        from ..medqa.need_flow import need_query
        text=need_query(result['tasks'][0],[{'id':'C1','text':'Synthetic adult NOT condition.'}])
        self.assertIn('verify synthetic relation K1',text)
        self.assertNotIn('Which relation',text)

    def test_covered_memory_actually_removes_search_and_reaches_reasoner(self):
        memory=self.memory();self.seed(memory);agents=NeedAgents()
        result=self.run_case(agents,memory=memory)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['retrieval_count'],0)
        self.assertEqual(result['metrics']['model_calls'],3)
        self.assertEqual(result['metrics']['memory_only_supported_tasks'],1)
        self.assertTrue(result['reused_memory_ids'])
        self.assertEqual(result['task_results']['K1']['support'],'supported')
        self.assertEqual(result['task_results']['K1']['lifecycle'],'completed')
        self.assertEqual(result['metrics']['memory_funnel']['investigator_accepted']['evidence'],1)
        self.assertEqual(result['metrics']['memory_evidence_presented'],1)
        self.assertTrue(agents.calls[-1][1]['evidence_pool'])
        self.assertIn('need_status',agents.calls[-1][1])

    def test_only_uncovered_need_is_searched(self):
        memory=self.memory();self.seed(memory)
        result=self.run_case(NeedAgents(two=True),memory=memory)
        searches=[e for e in result['events'] if e.get('tool')=='search_textbooks']
        self.assertEqual([e['knowledge_task_ids'] for e in searches],[['K2']],result)
        self.assertEqual(result['metrics']['memory_covered_needs'],1)
        self.assertEqual(result['metrics']['model_calls'],3)

    def test_overquery_for_covered_need_is_suppressed_by_runtime(self):
        memory=self.memory();self.seed(memory)
        result=self.run_case(NeedAgents(overquery=True),memory=memory)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['queries_skipped_covered'],1)
        self.assertEqual(result['metrics']['retrieval_count'],0)

    def test_inapplicable_memory_does_not_count_when_same_source_is_retrieved(self):
        memory=self.memory();self.seed(memory)
        result=self.run_case(NeedAgents(reject=True),memory=memory)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['retrieval_count'],1)
        self.assertEqual(result['reused_memory_ids'],[])
        self.assertEqual(result['metrics']['memory_funnel']['cited']['evidence'],0)
        self.assertEqual(result['metrics']['memory_funnel']['investigator_accepted']['evidence'],0)

    def test_rejected_memory_is_not_sent_to_reasoner_when_search_is_empty(self):
        memory=self.memory();self.seed(memory);agents=NeedAgents(reject=True)
        result=self.run_case(agents,memory=memory,books=Books(empty_until=100))
        self.assertEqual(agents.calls[-1][1].get('evidence_pool',[]),[])
        self.assertEqual(result['reused_memory_ids'],[])
        self.assertEqual(result['metrics']['memory_written'],0)

    def test_unresolved_candidate_same_source_is_not_counted_delivered(self):
        memory=self.memory();self.seed(memory)
        with patch.object(memory,'resolve',return_value=[]):
            result=self.run_case(memory=memory)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['memory_source_valid'],0)
        self.assertEqual(result['metrics']['memory_funnel']['delivered']['evidence'],0)

    def test_partial_review_returns_to_investigator_and_can_change_answer(self):
        agents=NeedAgents(follow=True)
        result=self.run_case(agents,books=NewBooks())
        self.assertEqual(result['answer'],'B',result)
        self.assertEqual([r for r,_,_ in agents.calls],[PLANNER,INVESTIGATOR,REASONER,INVESTIGATOR,REASONER])
        self.assertEqual(result['metrics']['investigation_rounds'],2)
        self.assertEqual(result['metrics']['retrieval_count'],2)
        self.assertGreater(result['metrics']['followup_new_visible_chars'],0)
        self.assertTrue(any('Distinct synthetic excerpt' in d['text'] for d in agents.calls[-1][1]['evidence_pool']))
        self.assertEqual(agents.calls[3][1]['active_need_ids'],['K1'])

    def test_insufficient_budget_preserves_answer_without_unfinishable_return(self):
        agents=NeedAgents(follow=True)
        result=self.run_case(agents,books=NewBooks(),max_calls=4)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['model_calls'],3)
        self.assertEqual(result['task_results']['K1']['support'],'partial')
        self.assertTrue(any(e.get('reason')=='investigator_and_reasoner_budget_unavailable' for e in result['events']))

    def test_no_novel_evidence_does_not_spend_second_reasoner_call(self):
        result=self.run_case(NeedAgents(follow=True),books=Books())
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['model_calls'],4)
        self.assertEqual(result['followup_result']['reason'],'followup_no_new_visible_evidence')

    def test_preserved_answer_after_unreviewed_investigation_does_not_promote_old_review(self):
        class PartlySupported(NeedAgents):
            def call(self,role,p,mode):
                output,usage=super().call(role,p,mode)
                if role==REASONER and output.get('next_action')=='investigate':
                    output['need_reviews'][0]['verdict']='supported'
                    output['investigation_request']['need_ids']=['K2']
                return output,usage
        result=self.run_case(PartlySupported(two=True,follow=True),memory=self.memory())
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['model_calls'],4)
        self.assertEqual(result['metrics']['memory_written'],0)

    def test_malformed_investigation_never_falls_back_to_unrequested_search(self):
        books=Books();result=self.run_case(NeedAgents(invalid=True),books=books)
        self.assertEqual(result['status'],'protocol_error',result)
        self.assertEqual(books.calls,0)

    def test_bad_reasoner_review_preserves_choice_but_writes_no_memory(self):
        class Invalid(NeedAgents):
            def call(self,role,p,mode):
                output,usage=super().call(role,p,mode)
                if role==REASONER:output['need_reviews'][0]['citation_indices']=[[0]]
                return output,usage
        result=self.run_case(Invalid(),memory=self.memory())
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['needs_review_errors'],1)
        self.assertEqual(result['metrics']['memory_written'],0)
        self.assertNotEqual(result['task_results']['K1']['support'],'supported')

    def test_memory_units_are_need_indexed_with_provenance_not_whole_case_index(self):
        memory=self.memory();self.seed(memory)
        with sqlite3.connect(self.root/'memory.sqlite3') as db:
            db.row_factory=sqlite3.Row
            row=dict(db.execute('SELECT * FROM memory').fetchone())
        content=json.loads(row['content'])
        self.assertEqual(row['task_topic'],'verify synthetic relation K1')
        self.assertEqual(content['index_schema'],'need-memory-v1')
        self.assertEqual(content['origin_question_id'],'first')
        self.assertTrue(row['created_at']);self.assertTrue(row['source_agent']);self.assertTrue(row['id'])
        text=embedding_text({'kind':row['kind'],'topic':row['task_topic'],'summary':row['summary'],'content':content})
        self.assertNotIn('Which relation applies?',text)
        self.assertIn(Books().doc['text'],text)

    def test_ten_sequential_tasks_reuse_persisted_evidence(self):
        memory=self.memory();rows=[]
        for index in range(10):
            self.q=Question('q'+str(index),self.q.question,self.q.options)
            result=self.run_case(memory=memory);rows.append(result)
            self.assertEqual(result['answer'],'A',result)
            self.assertEqual(result['metrics']['model_calls'],3)
            self.assertEqual(result['metrics']['memory_write_errors'],0)
        self.assertEqual(sum(r['metrics']['retrieval_count'] for r in rows),1)
        self.assertEqual(sum(r['metrics']['memory_only_supported_tasks'] for r in rows),9)

    def test_text_and_structured_share_need_policy_but_text_bypasses_acp(self):
        memory=self.memory();self.seed(memory)
        with patch('LangChain.medqa.runtime.ACPSession',side_effect=AssertionError('Text entered ACP')):
            result=self.run_case(memory=memory,mode='text')
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['acp_frames'],0)
        self.assertEqual(result['metrics']['retrieval_count'],0)
        event=next(e for e in result['events'] if e.get('agent')==REASONER)
        self.assertIn('调查覆盖判断为covered',event['model_input'])

    def test_summary_counts_presented_not_coarse_candidates(self):
        memory=self.memory();self.seed(memory)
        result=self.run_case(NeedAgents(two=True),memory=memory)
        summary=cli.summarize([{**result,'correct':True}])
        self.assertEqual(summary['memory_presented'],1)
        self.assertEqual(summary['memory_coarse_candidate_occurrences'],2)
        self.assertEqual(summary['memory_only_supported_tasks'],0)

    def test_real_chat_adapter_uses_need_prompts_and_exact_logged_input(self):
        model=NeedChat()
        agents=ChatRoles(model,output_policy='decoupled',memory_selection='needs',workspace_policy=POLICY,workflow='need_first')
        result=self.run_case(agents)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['model_calls'],3)
        events=[e for e in result['events'] if 'model_input' in e]
        for event,messages in zip(events,model.messages):
            self.assertEqual(event['model_input'],messages[1].content)
        self.assertIn('此阶段尚未检索历史记忆',model.messages[0][0].content)
        self.assertIn('need_updates',model.messages[1][0].content)
        self.assertIn('need_reviews',model.messages[2][0].content)
        self.assertIn('need_status',json.loads(model.messages[2][1].content))
        runtime=Runtime(agents,Books(),output_policy='decoupled',memory_policy='basic',memory_selection='needs',workflow='need_first')
        self.assertEqual(runtime.handshake()['memory_selection']['planner_limit'],0)
        self.assertIsNone(runtime.handshake()['memory_selection']['weights'])
        self.assertIn('assess_memory_coverage',runtime.handshake()['agents'][1]['capabilities'])

    def test_crash_replay_keeps_need_states_and_does_not_recall_model(self):
        benchmark=self.root/'benchmark.json';out=self.root/'run'
        benchmark.write_text(json.dumps({'medqa':{'first':{**self.q.public(),'answer':'A'}}}))
        books=Books();books.warmup=books.close=lambda:None
        embedding=SimpleNamespace(_model=object(),cache_key='fixture',cache_store=None)
        argv=['medqa','run','--benchmark',str(benchmark),'--output',str(out),'--split','all','--limit','1',
            '--memory','off','--output-policy','decoupled','--memory-policy','basic','--memory-selection','needs',
            '--workspace-policy',POLICY,'--workflow','need_first','--max-calls','7']
        first=NeedChat();second=NeedChat();original=DurableTools.execute
        def cut_after_search(journal,kind,request,callback):
            value=original(journal,kind,request,callback)
            if kind=='search_textbooks':raise PowerCut()
            return value
        def run(model,resume=False):
            agents=ChatRoles(model,output_policy='decoupled',memory_selection='needs',workspace_policy=POLICY,workflow='need_first')
            with (patch('sys.argv',argv+(['--resume'] if resume else [])),patch.object(cli,'books',return_value=books),
                  patch.object(cli,'create_model',return_value=(agents,{'model':'fixture'})),
                  patch('LangChain.memory.embedding.QwenEmbeddingService',return_value=embedding),patch('builtins.print')):
                cli.main()
        with patch.object(DurableTools,'execute',cut_after_search),self.assertRaises(PowerCut):run(first)
        run(second,resume=True)
        self.assertEqual(len(first.messages),2);self.assertEqual(len(second.messages),1)
        result=json.loads((out/'structured_off.jsonl').read_text())
        self.assertEqual(result['answer'],'A')
        self.assertEqual(result['task_results']['K1']['support'],'supported')
        self.assertEqual(result['metrics']['checkpoint_replayed_calls'],2)
        self.assertEqual(result['metrics']['total_tokens'],360)
        self.assertEqual(books.calls,1)

    def test_reasoner_reopens_provisionally_covered_memory(self):
        memory=self.memory();self.seed(memory)
        result=self.run_case(NeedAgents(follow=True),memory=memory,books=NewBooks())
        self.assertEqual(result['answer'],'B',result)
        self.assertEqual(result['metrics']['retrieval_count'],1)
        self.assertEqual(result['metrics']['model_calls'],5)

    def test_round_cap_prevents_unbounded_investigation(self):
        result=self.run_case(NeedAgents(follow=True),max_investigation_rounds=1)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(result['metrics']['model_calls'],3)
        self.assertTrue(any(e.get('reason')=='investigation_round_limit' for e in result['events']))


class SharedSourceFragmentsTest(unittest.TestCase):
    """Real service → source cache → bounded views → two downstream roles."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.memory=SharedMemory(Path(self.temp.name)/'memory.sqlite3','fragments','fixture')
        self.addCleanup(self.memory.close)
        self.quotes=[f'Synthetic {name} relation fragment is supported only within its declared scope.'
                     for name in ('alpha','bravo','charlie','delta')]
        self.text=('Unrelated context. '*100)+(' Background content. '*100).join(self.quotes)
        self.books=Books();self.books.doc.update(text=self.text,sha256=digest(self.text))
        self.q=Question('new','Synthetic adult NOT condition. Which relation applies?',{k:k for k in 'ABCD'})
        self.ids=[]
        for index in range(2):
            unit={'kind':'evidence','source_agent':REASONER,'topic':f'verify synthetic relation K{index+1}',
                  'summary':'\n'.join(self.quotes[index*2:index*2+2]),'document':deepcopy(self.books.doc),
                  'content':{**{k:self.books.doc[k] for k in ('id','title','sha256','corpus_version')},
                      'task_group':'g','origin_question_id':'old','index_schema':'need-memory-v1',
                      'excerpts':self.quotes[index*2:index*2+2]}}
            self.ids.append(self.memory.put_unit(unit,[1.,0.,0.,0.] if index==0 else [0.,1.,0.,0.],'seed'))

    def run_case(self,mode='structured',agents=None,**kwargs):
        trace_id=kwargs.pop('trace_id',None)
        class NeedEmbedding(Embeddings):
            def embed_text(self,text):return [0.,1.,0.,0.] if 'relation K2' in text else [1.,0.,0.,0.]
        quotes=self.quotes
        class Covered(NeedAgents):
            def __init__(self):super().__init__(two=True)
            def call(self,role,p,mode):
                output,usage=super().call(role,p,mode)
                if role==INVESTIGATOR:
                    for quote in quotes:
                        if not any(quote in d['text'] for d in p.get('evidence_pool',[])):
                            raise AssertionError('selected memory fragment was lost before Investigator')
                    output['action']='finish';output.pop('search',None)
                    output['need_updates']=[{'need_id':t['id'],'coverage':'covered',
                        'evidence_ids':[p['evidence_pool'][0]['id']],'condition_verdict':'matched',
                        'reason':'both actual visible source relations are present'} for t in p['knowledge_tasks']]
                if role==REASONER:
                    for quote in quotes:
                        if not any(quote in d['text'] for d in p.get('evidence_pool',[])):
                            raise AssertionError('selected memory fragment was lost before Reasoner')
                    output['citations']=[{'evidence_id':p['evidence_pool'][0]['id'],'quote':q,'support':'supported'} for q in quotes]
                    output['need_reviews']=[{'task_id':t['id'],'verdict':'supported','condition_verdict':'matched',
                        'citation_indices':[i*2,i*2+1],'reason':'visible relation and scope match'}
                        for i,t in enumerate(p['knowledge_tasks'])]
                return output,usage
        agents=agents or Covered()
        result=Runtime(agents,self.books,memory=self.memory,embedder=NeedEmbedding(),memory_write=False,
            output_policy='decoupled',memory_policy='basic',memory_selection='needs',workflow='need_first',
            workspace_policy=POLICY,task_group='g',mode=mode,max_reads=1,memory_limit=1,**kwargs).run(self.q,trace_id=trace_id)
        return result,agents

    def test_recall_preserves_same_source_units_when_requested(self):
        # Equal relevance of both units; ordinary legacy recall remains source-deduplicated.
        args=([1.,1.,0.,0.],'new',self.books.version)
        legacy=self.memory.recall(*args,threshold=.5,limit=10,task_group='g',kinds=['evidence'])
        fragments=self.memory.recall(*args,threshold=.5,limit=10,task_group='g',kinds=['evidence'],preserve_fragments=True)
        self.assertEqual(len(legacy),1)
        self.assertEqual({h['id'] for h in fragments},set(self.ids))

    def test_exact_knowledge_deduplicates_but_provenance_and_distinct_quotes_survive(self):
        doc=self.books.doc
        unit={'kind':'evidence','source_agent':REASONER,'topic':'stable relation',
              'summary':self.quotes[0],'document':deepcopy(doc),
              'content':{**{k:doc[k] for k in ('id','title','sha256','corpus_version')},
                  'task_group':'g','origin_question_id':'first','index_schema':'need-memory-v1',
                  'need_goal':'stable relation','conditions':['case one'],
                  'conditions_role':'origin_context_only; recheck_for_each_task','excerpts':[self.quotes[0]]}}
        first=self.memory.put_unit(unit,[1.,0.,0.,0.],'trace-one')
        unit['content'].update(origin_question_id='second',conditions=['case two'])
        second=self.memory.put_unit(unit,[1.,0.,0.,0.],'trace-two')
        self.assertEqual(first,second)
        self.assertEqual(second,self.memory.put_unit(unit,[1.,0.,0.,0.],'trace-two'))
        unit['content']['excerpts']=[self.quotes[1]]
        self.assertNotEqual(first,self.memory.put_unit(unit,[1.,0.,0.,0.],'trace-three'))
        with sqlite3.connect(Path(self.temp.name)/'memory.sqlite3') as db:
            origins=db.execute('SELECT context FROM memory_origin WHERE memory_id=?',(first,)).fetchall()
        self.assertEqual(len(origins),2)
        self.assertEqual({json.loads(r[0])['origin_question_id'] for r in origins},{'first','second'})

    def test_another_quote_from_same_source_is_not_fragment_reuse(self):
        doc=self.books.doc;start=doc['text'].index(self.quotes[0])
        fragments={'m':{'source_id':doc['id'],'source_sha256':doc['sha256'],
                         'spans':[{'start':start,'end':start+len(self.quotes[0])}]}}
        def check(quote):
            citation={'evidence_id':doc['id'],'quote':quote,'visible':True,'source_verified':True}
            result={'evidence_result':{'citations':[citation]}}
            reviews=[{'task_id':'K1','verdict':'supported','condition_verdict':'matched','support_quotes':[citation]}]
            return fragment_citations(fragments,{'m'},result,{doc['id']:doc},reviews)
        self.assertEqual(check(self.quotes[1]),(set(),set()))
        self.assertEqual(check(self.quotes[0]),({'m'},{'K1'}))
        fragments['m']['source_sha256']='changed'
        self.assertEqual(check(self.quotes[0]),(set(),set()))

    def test_memory_id_in_source_field_maps_end_to_end_in_both_modes(self):
        _,agents=self.run_case()
        original=agents.call
        def alias_call(role,p,mode):
            output,usage=original(role,p,mode)
            if role==INVESTIGATOR:
                for update in output['need_updates']:update['evidence_ids']=[self.ids[0]]
            return output,usage
        agents.call=alias_call
        for mode in ('text','structured'):
            result,_=self.run_case(mode=mode,agents=agents)
            self.assertEqual(result['status'],'answered',result)
            self.assertEqual(result['metrics']['retrieval_count'],0)
            self.assertEqual(set(result['reused_memory_ids']),set(self.ids))
            self.assertEqual(result['metrics']['memory_supported_needs'],2)
            self.assertTrue(any(e.get('tool')=='memory_source_alias_mapping' for e in result['events']))

    def test_two_needs_four_quotes_one_source_one_read_in_both_modes(self):
        for mode in ('structured','text'):
            self.books.stats['source_reads']=0
            result,agents=self.run_case(mode)
            self.assertEqual(result['answer'],'A',result)
            self.assertEqual(result['metrics']['model_calls'],3)
            self.assertEqual(result['metrics']['retrieval_count'],0)
            self.assertEqual(self.books.stats['source_reads'],1)
            self.assertEqual(result['metrics']['document_reads'],1)
            self.assertEqual(result['metrics']['memory_source_check_attempts'],1)
            self.assertEqual(result['metrics']['memory_source_valid'],1)
            self.assertEqual(result['metrics']['memory_fragment_valid'],2)
            self.assertEqual(result['metrics']['memory_source_cache_hits'],1)
            self.assertEqual(result['metrics']['memory_read_errors'],0)
            summary=cli.summarize([{**result,'correct':True}])
            self.assertEqual(summary['memory_source_check_coverage'],1)
            self.assertEqual(summary['memory_source_valid_rate'],1)
            self.assertEqual(set(result['memory_fragments']),set(self.ids))
            for index,ident in enumerate(self.ids):
                link=result['memory_fragments'][ident]
                self.assertEqual(link['knowledge_task_ids'],[f'K{index+1}'])
                self.assertEqual([self.text[s['start']:s['end']] for s in link['spans']],self.quotes[index*2:index*2+2])
            for role,p,_ in agents.calls:
                if role!=PLANNER:
                    self.assertEqual(len(p['evidence_pool']),1)
                    for quote in self.quotes:self.assertIn(quote,p['evidence_pool'][0]['text'])
                if role==INVESTIGATOR:
                    self.assertEqual({h['id'] for h in p['memory_hits']},set(self.ids))
                    self.assertTrue(all(h['knowledge_task_ids'] for h in p['memory_hits']))

    def test_failed_source_is_not_read_again_for_second_memory(self):
        original=self.books.read
        def changed(ident):
            doc=original(ident);doc['text']+=' changed';doc['sha256']=digest(doc['text']);return doc
        self.books.read=changed
        self.books.search=lambda query:[]
        result,_=self.run_case(agents=NeedAgents(two=True))
        self.assertEqual(self.books.stats['source_reads'],1)
        self.assertEqual(result['metrics']['document_reads'],1)
        self.assertEqual(result['metrics']['memory_source_check_attempts'],1)
        self.assertEqual(result['metrics']['source_read_errors'],1)
        self.assertEqual(result['metrics']['memory_source_valid'],0)
        self.assertEqual(result['memory_fragments'],{})

    def test_missing_memory_reference_does_not_borrow_other_units_source(self):
        resolve=self.memory.resolve
        def omit_one(*args,**kwargs):return [r for r in resolve(*args,**kwargs) if r['id']!=self.ids[0]]
        with patch.object(self.memory,'resolve',side_effect=omit_one):
            result,_=self.run_case(agents=NeedAgents(two=True))
        self.assertNotIn(self.ids[0],result['memory_fragments'])
        self.assertIn(self.ids[1],result['memory_fragments'])
        self.assertEqual(self.books.stats['source_reads'],1)

    def test_followup_can_add_same_source_fragment_with_no_read_budget_left(self):
        # Simulate a later need-directed recall finding a second unit from a
        # source already consumed by the first round's only permitted read.
        recall=self.memory.recall;count=[0]
        def staged_recall(values,*args,**kwargs):
            count[0]+=1
            vector=[1.,0.,0.,0.] if count[0]==1 else [0.,1.,0.,0.]
            return recall(vector,*args,**kwargs)
        quotes=self.quotes
        class Later(NeedAgents):
            def __init__(self):super().__init__(follow=True)
            def call(self,role,p,mode):
                out,usage=super().call(role,p,mode)
                if role==INVESTIGATOR:
                    out['action']='finish';out.pop('search',None)
                    out['need_updates'][0].update(coverage='partial',evidence_ids=[p['evidence_pool'][0]['id']],
                                                  condition_verdict='matched')
                if role==REASONER:
                    quote=quotes[0] if self.reasoner_calls==1 else quotes[2]
                    out['citations']=[{'evidence_id':p['evidence_pool'][0]['id'],'quote':quote,'support':'partial'}]
                    out['need_reviews'][0]['citation_indices']=[0]
                    if out.get('investigation_request'):
                        out['investigation_request']['gap']='charlie relation is missing'
                        out['investigation_request']['expected_information']='charlie relation fragment'
                return out,usage
        with patch.object(self.memory,'recall',side_effect=staged_recall):
            result,_=self.run_case(agents=Later())
        self.assertEqual(result['answer'],'B',result)
        self.assertEqual(result['metrics']['model_calls'],5)
        self.assertEqual(result['metrics']['document_reads'],1)
        self.assertEqual(self.books.stats['source_reads'],1)
        self.assertEqual(set(result['memory_fragments']),set(self.ids))
        self.assertEqual(result['metrics']['memory_source_cache_hits'],1)

    def test_context_limit_defers_source_without_discarding_stored_fragment_links(self):
        result,_=self.run_case(agents=NeedAgents(two=True),max_evidence_chars=100)
        self.assertEqual(set(result['memory_fragments']),set(self.ids))
        self.assertIn(self.books.doc['id'],result['evidence_processing']['omitted'])
        self.assertFalse(result['evidence_processing']['views'])
        self.assertEqual(result['metrics']['memory_only_supported_tasks'],0)

    def test_committed_source_read_is_replayed_without_a_second_physical_read(self):
        folder=Path(self.temp.name)/'checkpoint';folder.mkdir()
        with Checkpoint(folder) as checkpoint:
            original=DurableTools.execute
            def cut(journal,kind,request,callback):
                value=original(journal,kind,request,callback)
                if kind=='read_source':raise PowerCut()
                return value
            with patch.object(DurableTools,'execute',cut),self.assertRaises(PowerCut):
                self.run_case(journal=DurableTools(checkpoint,'structured_on',0,max_reads=1),trace_id='fixed')
            result,_=self.run_case(journal=DurableTools(checkpoint,'structured_on',0,max_reads=1),trace_id='fixed')
            self.assertEqual(result['answer'],'A',result)
            self.assertEqual(self.books.stats['source_reads'],1)
            self.assertEqual(result['metrics']['memory_source_cache_hits'],1)
            self.assertEqual(set(result['memory_fragments']),set(self.ids))

    def test_serialized_chat_inputs_keep_both_needs_source_fragments(self):
        initial,_=self.run_case()
        outputs=[e['output'] for e in initial['events'] if e.get('agent') in {PLANNER,INVESTIGATOR,REASONER} and 'output' in e]
        class ReplayChat:
            def __init__(self):self.messages=[]
            def invoke(self,messages):
                output=outputs[len(self.messages)];self.messages.append(messages)
                return SimpleNamespace(content=json.dumps(output),response_metadata={'model':'fixture','finish_reason':'stop',
                    'token_usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})
        model=ReplayChat();self.books.stats['source_reads']=0
        agents=ChatRoles(model,output_policy='decoupled',memory_selection='needs',workflow='need_first',workspace_policy=POLICY)
        result,_=self.run_case(agents=agents)
        self.assertEqual(result['answer'],'A',result)
        self.assertEqual(self.books.stats['source_reads'],1)
        for messages in model.messages[1:]:
            payload=json.loads(messages[1].content)
            self.assertEqual(len(payload['evidence_pool']),1)
            for quote in self.quotes:self.assertIn(quote,payload['evidence_pool'][0]['text'])


class NeedContractsTest(unittest.TestCase):
    def setUp(self):self.doc=Books().doc

    def test_source_selection_is_global_round_robin_deduplicated_and_zero_safe(self):
        candidates={i:{'content':{'id':source,'sha256':source}} for i,source in
                    [('a','S1'),('b','S2'),('c','S3'),('d','S1')]}
        rankings={'K1':['a','b'],'K2':['c','d']}
        self.assertEqual(select_sources(rankings,candidates,2),['a','c','d'])
        self.assertEqual(select_sources(rankings,candidates,5),['a','c','b','d'])
        self.assertEqual(select_sources(rankings,candidates,0),[])
        self.assertEqual(select_sources(rankings,candidates,0,known_sources={('S1','S1')}),['a','d'])
        self.assertEqual(select_sources(rankings,candidates,2,fragment_limit=1),['a'])
        many={str(i):{'content':{'id':'S1','sha256':'S1'}} for i in range(20)}
        self.assertEqual(len(select_sources({'K1':list(many)},many,1)),9)

    def test_covered_requires_visible_source_and_matching_conditions(self):
        update={'need_id':'K1','coverage':'covered','evidence_ids':[self.doc['id']],
                'condition_verdict':'unknown','reason':'synthetic'}
        for condition,visible in [('unknown',{self.doc['id']:[]}),('matched',{})]:
            update['condition_verdict']=condition
            with self.assertRaises(ContractError):investigation_reviews([update],{'K1':{}},{self.doc['id']:self.doc},visible)

    def test_resolved_memory_alias_maps_only_to_its_actual_visible_source(self):
        update={'need_id':'K1','coverage':'partial','evidence_ids':['mem-known'],
                'condition_verdict':'unknown','reason':'source relation partially available'}
        docs={self.doc['id']:self.doc};visible={self.doc['id']:[]}
        aliases={'mem-known':self.doc['id']}
        result=investigation_reviews([update],{'K1':{}},docs,visible,memory_sources=aliases)
        self.assertEqual(result['K1']['coverage_evidence_ids'],[self.doc['id']])
        for shown,mapping in [(visible,{}),({},aliases),(visible,{'mem-known':'missing-source'})]:
            with self.assertRaises(ContractError):
                investigation_reviews([update],{'K1':{}},docs,shown,memory_sources=mapping)

    def test_need_reviews_are_atomic_including_control_validation(self):
        states=TaskStates([{'id':'K1'},{'id':'K2'}]);before=deepcopy(states.items)
        output={'citations':[{'evidence_id':self.doc['id'],'quote':self.doc['text']}],
                'need_reviews':[{'task_id':ident,'verdict':'supported','citation_indices':[0],
                    'condition_verdict':'matched','reason':'synthetic'} for ident in ('K1','K2')],
                'next_action':'investigate','investigation_request':{'need_ids':['K1'],
                    'gap':'already supported','expected_information':'irrelevant','may_change_answer':True}}
        with self.assertRaises(ContractError):
            reasoner_reviews(output,states,{self.doc['id']:self.doc},{self.doc['id']:[{'text':self.doc['text']}]})
        self.assertEqual(states.items,before);self.assertEqual(states.transitions,[])

    def test_memory_excerpt_keeps_raw_offsets_context_and_expandability(self):
        from ..medqa.evidence_processor import EvidenceProcessor, validate_view
        text='Unrelated preceding background. '*80+'Important synthetic scope relation applies only to adults.'+' Unrelated later background.'*80
        doc={**self.doc,'text':text,'sha256':digest(text)}
        quote='Important synthetic scope relation applies only to adults.'
        pool={doc['id']:doc}
        pack=EvidenceProcessor().process(pool,query='synthetic scope relation',protected={doc['id']:[quote]})
        shorter=compact_memory_view(pack,pool,{doc['id']:[quote]})
        validate_view(shorter,pool)
        self.assertIn(quote,shorter['views'][0]['text'])
        self.assertTrue(shorter['views'][0]['has_more'])
        self.assertLess(shorter['view_chars'],pack['view_chars'])
        self.assertEqual(doc['text'],text)

    def test_identical_raw_sources_with_different_fragments_do_not_alias_away_a_view(self):
        from ..medqa.evidence_processor import EvidenceProcessor,validate_view
        first='Distinct alpha source passage has its own scope.'
        second='Different beta source passage has another scope.'
        text=first+' Unrelated background.'*200+second
        docs={ident:{**self.doc,'id':ident,'text':text,'sha256':digest(text)} for ident in ('A','B')}
        pack=EvidenceProcessor().process(docs,query='relation',excerpt_only={'A':[first],'B':[second]})
        validate_view(pack,docs)
        self.assertEqual(len(pack['views']),2)
        views={d['id']:d['text'] for d in pack['views']}
        self.assertIn(first,views['A']);self.assertIn(second,views['B'])


if __name__=='__main__':unittest.main()
