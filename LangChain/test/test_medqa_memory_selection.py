"""Selection counterexamples, real IPC lifecycle and bounded cost regressions."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .test_medqa import Books,Embeddings
from .test_medqa_memory_units import MinimalAgents
from ..medqa.memory_selection import select,previews,prefilter,applicability
from ..medqa.grounding import validate_plan,ContractError
from ..medqa.memory import SharedMemory
from ..medqa.runtime import Runtime,compact,ChatRoles
from ..medqa.data import Question
from ..agents.medqa_prompts import PLANNER,INVESTIGATOR,REASONER
from ..agents.medqa_baseline_prompts import BASELINE_ROLE_PROMPTS


def hit(ident,score,topic,summary,kind='evidence'):
    return {'id':ident,'similarity':score,'task_topic':topic,'summary':summary,'kind':kind,
            'content':{'id':ident,'sha256':ident}}


def read(ident,needs):
    return {'memory_id':ident,'decision':'read','reason':'potential use','knowledge_task_ids':needs}


class SelectionTest(unittest.TestCase):
    def test_lower_semantic_rank_but_matching_goal_enters_final_selection(self):
        hits={h['id']:h for h in [
            hit('general',.55,'Mechanism of salivary secretion','A general receptor mechanism'),
            hit('direct',.509,'Uses of a drug','A previously sourced treatment relation'),
            hit('complement',.505,'Indications for treatment','A distinct applicable treatment relation')]}
        selected,records=select('Which treatment should be used?',hits,[read(i,['K1']) for i in hits],['K1'],2)
        self.assertEqual(selected,['direct','complement'])
        self.assertEqual(records[0]['T'],1)
        self.assertEqual(records[1]['C'],0)  # marginal rather than fixed coverage

    def test_marginal_coverage_and_redundancy_avoid_duplicate_selection(self):
        hits={h['id']:h for h in [hit('a',.9,'Treatment','alpha beta gamma delta'),
            hit('duplicate',.89,'Treatment','alpha beta gamma delta'),
            hit('new',.85,'Treatment','epsilon zeta other information')]}
        selected,_=select('Treatment?',hits,[read('a',['K1']),read('duplicate',['K1']),read('new',['K2'])],['K1','K2'],2)
        self.assertEqual(selected,['a','new'])

    def test_strategy_coverage_does_not_consume_evidence_coverage(self):
        hits={'s':hit('s',.9,'Treatment','query hint','strategy'),
              'e':hit('e',.8,'Treatment','source excerpt')}
        selected,records=select('Treatment?',hits,[read('s',['K1']),read('e',['K1'])],['K1'],2)
        self.assertEqual(selected,['s','e'])
        self.assertEqual([r['C'] for r in records],[1,1])

    def test_no_type_quota_and_no_forced_selection(self):
        hits={h['id']:h for h in [hit('a',.8,'Treatment','first'),hit('b',.7,'Treatment','second'),
            hit('strategy',.9,'Research study design','different topic','strategy')]}
        assessments=[read('a',['K1']),read('b',['K1']),{'memory_id':'strategy','decision':'reject','reason':'unrelated','knowledge_task_ids':[]}]
        selected,_=select('Treatment?',hits,assessments,['K1'],3)
        self.assertEqual(selected,['a','b'])
        self.assertEqual(select('Treatment?',hits,[],['K1'])[0],[])

    def test_aliases_of_the_same_source_content_do_not_take_two_slots(self):
        a=hit('a',.9,'Treatment','one source')
        b=hit('alias',.89,'Treatment','different metadata')
        b['content']['sha256']=a['content']['sha256']
        chosen,_=select('Treatment?',{'a':a,'alias':b},[read('a',['K1']),read('alias',['K2'])],['K1','K2'],3)
        self.assertEqual(chosen,['a'])

    def test_coarse_candidates_cannot_all_enter_planner_preview(self):
        hits=[hit('mem-'+str(i).zfill(32),.8,'a long task topic'*30,'摘要'*400) for i in range(10)]
        preview=previews(hits,2400,compact)
        self.assertLessEqual(len(preview),3)
        self.assertLessEqual(len(compact(preview)),1000)
        self.assertTrue(all(set(h)<= {'id','kind','summary','origin_conditions'} for h in preview))

    def test_program_prefilter_reranks_before_capping_and_removes_duplicates(self):
        a=hit('direct',.65,'Treatment','target treatment relation')
        alias=hit('alias',.64,'Treatment','duplicate source with different description')
        alias['content']['sha256']=a['content']['sha256']
        irrelevant=hit('research',.7,'Research study design','unrelated statistical design')
        selected,records=prefilter('Which treatment is indicated?',{x['id']:x for x in [irrelevant,alias,a]})
        self.assertEqual(selected,['direct'])
        self.assertEqual(next(r for r in records if r['memory_id']=='alias')['reason'],
                         'duplicate_source_or_high_lexical_overlap')

    def test_applicability_only_uses_explicit_rules_and_keeps_uncertainty(self):
        old=hit('old',.9,'Treatment','a reusable source')
        old['content'].update(conditions=['an infant in a previous task'],conditions_role='source_task_context')
        self.assertEqual(applicability('An adult needs treatment',old),(.5,'requires_current_task_review',False))
        old['content']['applicability']={'excluded_terms':['pregnant']}
        self.assertTrue(applicability('A pregnant adult needs treatment',old)[2])
        self.assertFalse(applicability('An adult is not pregnant and needs treatment',old)[2])
        old['content']['applicability']={'required_terms':['adult']}
        self.assertFalse(applicability('The age is not stated',old)[2])
        self.assertEqual(applicability('The age is not stated',old)[1],'required_terms_not_established')

    def test_prefilter_can_select_nothing(self):
        self.assertEqual(prefilter('Treatment?',{'low':hit('low',.1,'Research study design','unrelated')})[0],[])

    def test_sparse_assessment_is_opt_in_and_unknown_id_is_rejected(self):
        agents=MinimalAgents()
        plan,_=agents.call(PLANNER,{'question_id':'q','memory_hits':[]},'structured')
        validate_plan(plan,{'M1'},sparse_memories=True)
        with self.assertRaises(ContractError):validate_plan(plan,{'M1'})
        plan['memory_assessments']=[read('invented',['K1'])]
        with self.assertRaises(ContractError):validate_plan(plan,{'M1'},sparse_memories=True)

    def test_legacy_prompt_bytes_remain_unchanged(self):
        legacy=ChatRoles(object(),output_policy='decoupled')
        needs=ChatRoles(object(),output_policy='decoupled',memory_selection='needs')
        self.assertEqual(legacy.prompts,BASELINE_ROLE_PROMPTS)
        self.assertNotEqual(needs.prompts[PLANNER],legacy.prompts[PLANNER])
        self.assertEqual(needs.prompts[REASONER],legacy.prompts[REASONER])


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory=SharedMemory(Path(self.temp.name)/'memory.sqlite3','selection-fixture','fixture-model')
        self.addCleanup(self.memory.close)

    def run_case(self,ident,agents=None,**kwargs):
        return Runtime(agents or MinimalAgents(),Books(),memory=self.memory,embedder=Embeddings(),
            output_policy='decoupled',memory_policy='basic',memory_selection='needs',max_calls=4,
            task_group='g',**kwargs).run(Question(ident,'A synthetic treatment question?',{k:k for k in 'ABCD'}))

    def test_strategy_execution_requires_delivered_reference_and_an_executed_action(self):
        self.run_case('first')
        class UsingStrategy(MinimalAgents):
            def call(self,role,payload,mode):
                data,usage=super().call(role,payload,mode)
                if role==INVESTIGATOR:
                    ids=[h['id'] for h in payload['memory_hits'] if h.get('kind')=='strategy']
                    data['search']['queries'][0]['strategy_memory_ids']=ids
                return data,usage
        agents=UsingStrategy()
        with patch.object(self.memory,'recall',wraps=self.memory.recall) as recall:
            result=self.run_case('second',agents)
        self.assertEqual(recall.call_args.kwargs['limit'],10)
        self.assertFalse(recall.call_args.kwargs['diversify'])
        funnel=result['metrics']['memory_funnel']
        self.assertEqual(funnel['selected'],{'evidence':1,'strategy':1,'episode':1})
        self.assertEqual(funnel['executed']['strategy'],1)
        self.assertEqual(funnel['cited']['evidence'],1)
        self.assertEqual(result['metrics']['model_calls'],3)
        self.assertEqual(result['metrics']['state_transfers'],result['metrics']['state_vector_transfers']+1)
        self.assertEqual(result['metrics']['state_bytes'],result['metrics']['state_vector_transfers']*16)
        self.assertTrue(any(e.get('tool')=='memory_strategy_execution' for e in result['events']))
        investigator=next(p for role,p,_ in agents.calls if role==INVESTIGATOR)
        strategy=next(h for h in investigator['memory_hits'] if h.get('kind')=='strategy')
        self.assertNotIn('similarity',strategy)
        self.assertNotIn('created_at',strategy)
        planner=next(p for role,p,_ in agents.calls if role==PLANNER)
        preview=next(h for h in planner['memory_hits'] if h.get('kind')=='strategy')
        self.assertIn('initial fixture',preview['summary'])

    def test_invalid_strategy_reference_does_not_break_search_or_gain_credit(self):
        class BadReference(MinimalAgents):
            def call(self,role,payload,mode):
                data,usage=super().call(role,payload,mode)
                if role==INVESTIGATOR:data['search']['queries'][0]['strategy_memory_ids']=['invented']
                return data,usage
        result=self.run_case('first',BadReference())
        self.assertEqual(result['answer'],'A')
        self.assertEqual(result['metrics']['retrieval_count'],1)
        self.assertEqual(result['metrics']['memory_funnel']['executed']['strategy'],0)

    def test_ten_coarse_records_stay_internal_and_compact_choices_need_no_reasons(self):
        for i,word in enumerate(('alpha','beta','gamma','delta','epsilon','zeta','theta','kappa','lambda','omega')):
            self.memory.put_unit({'kind':'strategy','source_agent':'Investigator','topic':'Treatment',
                'summary':word+' useful query', 'content':{'corpus_version':Books.version,'task_group':'g',
                    'origin_question_id':'seed'+str(i),'queries':[{'query':word,'source_ids':[]}]}},
                [1.,0.,0.,0.],'seed'+str(i))
        class CompactPlanner(MinimalAgents):
            def call(self,role,payload,mode):
                data,usage=super().call(role,payload,mode)
                if role==PLANNER:
                    for item in data['memory_assessments']:
                        item.pop('decision');item.pop('reason')
                return data,usage
        agents=CompactPlanner();result=self.run_case('later',agents)
        planner=next(p for role,p,_ in agents.calls if role==PLANNER)
        self.assertEqual(result['metrics']['memory_candidates'],10)
        self.assertLessEqual(len(planner['memory_hits']),3)
        self.assertLessEqual(len(compact(planner['memory_hits'])),1000)
        self.assertEqual(result['answer'],'A')
        self.assertEqual(result['metrics']['model_calls'],3)
        audit=next(e for e in result['events'] if e.get('tool')=='memory_prefilter')
        self.assertEqual(len(audit['coarse_ids']),10)
        self.assertEqual(audit['presented_ids'],[h['id'] for h in planner['memory_hits']])

    def test_coarse_recall_is_top_score_without_forced_type_slots(self):
        common={'corpus_version':Books.version,'task_group':'g'}
        for i in range(11):
            unit={'kind':'strategy','source_agent':'Investigator','topic':'task '+str(i),'summary':'query '+str(i),
                  'content':{**common,'origin_question_id':'q'+str(i),'queries':[]}}
            self.memory.put_unit(unit,[1.,0.,0.,0.],'seed')
        self.memory.put_unit({'kind':'episode','source_agent':'Planner','topic':'other','summary':'lower score',
                'content':{**common,'origin_question_id':'other'}},[.8,.6,0.,0.],'seed')
        hits=self.memory.recall([1.,0.,0.,0.],'later',Books.version,limit=10,task_group='g',
                               kinds=['evidence','strategy','episode'],diversify=False)
        self.assertEqual(len(hits),10)
        self.assertTrue(all(h['kind']=='strategy' for h in hits))

    def test_strategy_never_gets_factual_citation_credit(self):
        self.run_case('first')
        agents=MinimalAgents(reuse=True)
        result=self.run_case('second',agents)
        reasoner=next(p for role,p,_ in agents.calls if role==REASONER)
        self.assertFalse(any(h.get('kind') in {'strategy','episode'} for h in reasoner['memory_hits']))
        self.assertEqual(result['metrics']['memory_funnel']['cited']['strategy'],0)
        self.assertEqual(result['metrics']['memory_funnel']['executed']['strategy'],0)

    def test_unexecuted_fallback_query_does_not_gain_strategy_credit(self):
        self.run_case('first')
        class FallbackOnly(MinimalAgents):
            def call(self,role,payload,mode):
                data,usage=super().call(role,payload,mode)
                if role==INVESTIGATOR:
                    ids=[h['id'] for h in payload['memory_hits'] if h.get('kind')=='strategy']
                    first=data['search']['queries'][0]
                    fallback=deepcopy(first)
                    fallback.update(id='Q2',query='unused fallback',run_if='empty_or_duplicate',after=first['id'],strategy_memory_ids=ids)
                    data['search']['queries'].append(fallback)
                return data,usage
        original=Books.search
        def fresh(book,query):
            docs=original(book,query)
            for doc in docs:
                doc['id']='New_'+doc['id']
                doc['text']+=' A distinct synthetic detail.'
                doc['sha256']=hashlib.sha256(doc['text'].encode()).hexdigest()
            return docs
        with patch.object(Books,'search',fresh):
            result=self.run_case('second',FallbackOnly())
        self.assertEqual(result['metrics']['retrieval_count'],1)
        self.assertEqual(result['metrics']['memory_funnel']['delivered']['strategy'],1)
        self.assertEqual(result['metrics']['memory_funnel']['executed']['strategy'],0)

    def test_ten_needs_tasks_keep_three_calls_and_bounded_memory_input(self):
        for n in range(10):
            agents=MinimalAgents(reuse=True)
            result=self.run_case('round-'+str(n),agents)
            self.assertEqual(result['metrics']['model_calls'],3)
            self.assertLessEqual(sum(result['metrics']['memory_funnel']['selected'].values()),3)
            planner=next(p for role,p,_ in agents.calls if role==PLANNER)
            self.assertLessEqual(len(planner['memory_hits']),3)
            self.assertLessEqual(len(compact(planner['memory_hits'])),1000)
            if n:self.assertEqual(result['metrics']['memory_funnel']['cited']['evidence'],1)


if __name__=='__main__':unittest.main()
