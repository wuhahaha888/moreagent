"""Recovery tests use local fixtures only, without API access."""
import json
from pathlib import Path
import tempfile
import unittest

from LangChain.medqa import cli
from LangChain.medqa.checkpoint import Checkpoint, CheckpointError, DurableCalls, DurableTools
from LangChain.medqa.data import Dataset
from LangChain.medqa.runtime import Runtime
from LangChain.test.test_medqa import Agents, Books
from scripts.resume_medqa_failed import plan_recovery, reopen_failure, run


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.benchmark = self.root/'questions.json'
        self.benchmark.write_text(json.dumps({'medqa':{
            q:{'question':'fixture','options':{k:k for k in 'ABCD'},'answer':'A'} for q in ('q0','q1')}}))
        self.data = Dataset(self.benchmark)
        self.questions = self.data.select(split='all')
        self.out = self.root/'run'
        self.out.mkdir()
        self.store = Checkpoint(self.out)
        self.addCleanup(self.store.close)
        self.store.initialize({'settings':{}},[['text','on'],['structured','off'],['structured','on'],['text','off']],resume=False)
        self.books = Books()
        for condition in ('text_on','structured_off','structured_on'):
            for position,q in enumerate(self.questions):
                row = self.store.begin(condition,position,q.id)
                result = Runtime(Agents(),self.books,mode=condition.rsplit('_',1)[0],max_calls=4).run(q,trace_id=row['trace'])
                result['correct'] = self.data.grade(result)
                self.store.save_result(condition,position,result,[])
            self.store.apply_memory(condition,None)
        class Failed:
            def call(self,*args,**kwargs):
                raise type('APIConnectionError',(Exception,),{})()
        row = self.store.begin('text_off',0,'q0')
        calls = DurableCalls(Failed(),self.store,'text_off',0,max_calls=4)
        tools = DurableTools(self.store,'text_off',0)
        result = Runtime(calls,self.books,mode='text',max_calls=4,journal=tools,
                         output_policy='decoupled',memory_policy='basic',memory_selection='needs').run(self.questions[0],trace_id=row['trace'])
        result['correct'] = False
        result['metrics']['checkpoint_unknown_calls'] = 1
        self.store.save_result('text_off',0,result,[])
        self.store.apply_memory('text_off',None)

    def test_plan_preserves_original_result_and_trace(self):
        plan = plan_recovery(self.store,'text_off',self.questions)
        self.assertEqual(plan['question_ids'],['q0','q1'])
        before = self.store.question('text_off',0)
        original_result,trace = before['result'],before['trace']
        reopen_failure(self.store,plan)
        self.assertEqual(self.store.question('text_off',0)['trace'],trace)
        self.assertEqual(self.store.unknown_calls('text_off'),1)
        self.assertEqual(self.store.meta('bounded_recovery_history')[0]['failed_row']['result'],original_result)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM control_states').fetchone()[0],0)

    def test_valid_answer_cannot_be_reopened(self):
        result=json.loads(self.store.question('text_off',0)['result']); result.update(answer='A',status='answered')
        self.store.save_result('text_off',0,result,[]); self.store.apply_memory('text_off',None)
        with self.assertRaises(CheckpointError): plan_recovery(self.store,'text_off',self.questions)

    def test_memory_enabled_condition_rejected(self):
        with self.assertRaises(CheckpointError): plan_recovery(self.store,'text_on',self.questions)

    def test_other_unknown_request_blocks_recovery(self):
        with self.store.db:
            self.store.db.execute("INSERT INTO calls VALUES ('structured_off',0,1,1,'hash','Planner','pending',NULL,NULL,NULL)")
        with self.assertRaises(CheckpointError): plan_recovery(self.store,'text_off',self.questions)

    def test_tool_history_cannot_be_discarded(self):
        tools=DurableTools(self.store,'text_off',0)
        tools.execute('fixture',{},lambda:{'value':1})
        with self.assertRaises(CheckpointError): plan_recovery(self.store,'text_off',self.questions)

    def test_bounded_run_only_requests_two_tasks_and_keeps_unknown_cost(self):
        plan=plan_recovery(self.store,'text_off',self.questions)
        unaffected={c:self.store.results(c) for c in ('text_on','structured_on','structured_off')}
        reopen_failure(self.store,plan)
        _,args=cli.parse_args(['compare','--all','--max-calls','4','--memory-policy','basic','--memory-selection','needs'])
        agents=Agents()
        run(args,self.data,self.books,agents,self.store,self.out,self.questions,plan)
        self.assertEqual(len(agents.calls),6)
        self.assertTrue(all(r['correct'] for r in self.store.results('text_off')))
        self.assertEqual(self.store.unknown_calls('text_off'),1)
        self.assertEqual(self.store.db.execute("SELECT state FROM calls WHERE condition='text_off' AND ordinal=1 AND attempt=1").fetchone()[0],'abandoned')
        self.assertEqual(self.store.results('text_off')[0]['metrics']['model_calls'],4)
        self.assertEqual(self.store.results('text_off')[0]['metrics']['checkpoint_request_attempts'],4)
        self.assertEqual(json.loads((self.out/'report.json').read_text())['text_off']['usage_missing_calls'],1)
        self.assertFalse(json.loads((self.out/'comparison.json').read_text())['token_comparison_valid'])
        for c,rows in unaffected.items():self.assertEqual(self.store.results(c),rows)


if __name__=='__main__':unittest.main()
