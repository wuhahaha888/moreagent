"""Protocol/transport and actual model-input tests; all models are local fixtures."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .test_medqa import Agents, Books, Embeddings
from .test_medqa_memory_units import MinimalAgents
from ..agents.medqa_prompts import PLANNER,INVESTIGATOR,REASONER
from ..medqa.acp import Session, Actor, Opcode, Kind, Status, ACPError, HEADER, pack, unpack
from ..medqa.data import Question
from ..medqa.handoff import describe_output
from ..medqa.grounding import ContractError
from ..medqa.runtime import Runtime,ChatRoles
from ..medqa.cli import summarize


class ACPTest(unittest.TestCase):
    def setUp(self):
        self.session=Session(17,memory_enabled=False)

    def test_binary_header_and_position_parameters_roundtrip(self):
        packet=pack(Kind.REQUEST,Actor.INVESTIGATOR,Actor.TEXTBOOKS,Opcode.SEARCH,17,42,(101,4))
        self.assertEqual(len(packet),26)
        self.assertEqual(unpack(packet),(Kind.REQUEST,Actor.INVESTIGATOR,Actor.TEXTBOOKS,Opcode.SEARCH,17,42,(101,4)))
        for bad in [packet[:-1],packet+b'x',b'',bytes([2])+packet[1:]]:
            with self.assertRaises(ACPError):unpack(bad)
        with self.assertRaises(ACPError):pack(1,2,4,10,17,1,(True,4))

    def test_handshake_capabilities_control_real_dispatch(self):
        session=self.session;called=[]
        session.bind(Actor.TEXTBOOKS,Opcode.SEARCH,lambda q,k:called.append((q,k)) or ['source'])
        ref=session.put('query','fixture query')
        result=session.call(Actor.INVESTIGATOR,Actor.TEXTBOOKS,Opcode.SEARCH,(ref,4))
        self.assertEqual(result,['source']);self.assertEqual(called,[('fixture query',4)])
        self.assertGreater(session.metrics['acp_handshake_bytes'],0)
        self.assertNotIn(int(Actor.MEMORY),session.description()['ready'])
        session.ready.remove(Actor.TEXTBOOKS)
        with self.assertRaises(ACPError):session.call(2,4,Opcode.SEARCH,(ref,4))
        self.assertEqual(len(called),1)
        kind,src,dst,op,task,request,args=unpack(session.frames[-1])
        self.assertEqual(kind,Kind.ERROR);self.assertEqual(args[0],Status.INVALID)

    def test_unknown_capability_wrong_caller_and_reference_fail_before_tool(self):
        called=[];session=self.session
        session.bind(4,Opcode.SEARCH,lambda q,k:called.append(q))
        ref=session.put('query','fixture')
        for src,values in [(Actor.PLANNER,(ref,4)),(Actor.INVESTIGATOR,(ref,)),
                           (Actor.INVESTIGATOR,(ref,0)),(Actor.INVESTIGATOR,(99999,4)),
                           (Actor.INVESTIGATOR,(session.put('source','fixture'),4))]:
            with self.subTest(src=src,values=values),self.assertRaises(ACPError):session.call(src,4,Opcode.SEARCH,values)
        self.assertEqual(called,[])
        with self.assertRaises(ACPError):session.bind(4,Opcode.ANSWER,lambda:None)
        with self.assertRaises(ACPError):session.negotiate(Actor.RUNTIME,versions=(2,))

    def test_references_are_immutable_typed_and_task_scoped(self):
        session=self.session;original={'value':[1]};ref=session.put('context',original)
        original['value'].append(2);value=session.get(ref,'context');value['value'].append(3)
        self.assertEqual(session.get(ref,'context'),{'value':[1]})
        with self.assertRaises(ACPError):session.get(ref,'task')
        with self.assertRaises(ACPError):Session(18).get(ref,'context')
        session.bind(4,Opcode.SEARCH,lambda q,k:[])
        with self.assertRaises(ACPError):session.receive(pack(1,2,4,10,18,1,(ref,4)))

    def test_vector_handle_is_consumed_by_registered_receiver(self):
        import numpy as np
        session=Session(17,embedding_model='fixture-v1');captured=[]
        session.bind(Actor.MEMORY,Opcode.RECALL,lambda v,p:captured.append(v) or {'hits':[]})
        state=session.publish_vector([1.,.5,0.]);params=session.put('recall_parameters',{})
        self.assertEqual(state,session.publish_vector([1.,.5,0.]))
        self.assertEqual(session.call(0,5,Opcode.RECALL,(state,params)),{'hits':[]})
        self.assertEqual(captured[0].dtype,np.dtype('<f4'))
        self.assertEqual(captured[0].tolist(),[1.,.5,0.])
        self.assertEqual(session.metrics['state_handle_reads'],1)
        self.assertEqual(session.metrics['state_handle_bytes'],12)
        session.embedding_model='other-model'
        with self.assertRaises(ACPError):session.consume_vector(state)
        with self.assertRaises(ACPError):session.publish_vector([float('nan')])

    def test_tool_plan_batch_preserves_order_and_explicit_conditions(self):
        session=self.session;session.bind(Actor.RUNTIME,Opcode.BATCH,session.accept_batch)
        plan={'action':'tool_plan','search':{'gap':'fixture gap','queries':[
            {'id':'first','query':'query one','knowledge_task_ids':['K1'],'expected_information':'fixture','run_if':'always'},
            {'id':'backup','query':'query two','knowledge_task_ids':['K1'],'expected_information':'fixture',
             'run_if':'empty_or_duplicate','after':'first'}]}}
        self.assertEqual(session.map_tool_plan(plan),plan)
        self.assertEqual(session.metrics['acp_batches'],1)
        for change in [{'after':'missing'},{'id':'first'},{'run_if':'unknown'}]:
            invalid=deepcopy(plan);invalid['search']['queries'][1].update(change)
            with self.assertRaises(ACPError):session.map_tool_plan(invalid)


class ScriptedChat:
    """Capture what a real ChatRoles adapter sends, without a model endpoint."""
    def __init__(self):self.messages=[]
    def invoke(self,messages):
        self.messages.append(messages)
        result={'question_id':'q'}
        if len(self.messages)==1:
            result.update(key_conditions=[{'id':'C1','text':'NOT, adult, 10 mg, time limit'}],
                knowledge_tasks=[{'id':'K1','goal':'check synthetic scope','condition_ids':['C1'],'depends_on':[]}],
                criteria=['source support'],memory_assessments=[],
                handoff_text='请核实需求K1，保留NOT、成人、10 mg及时间限制；不要提前认定选项。')
        elif len(self.messages)==2:
            result.update(action='tool_plan',search={'gap':'synthetic gap','queries':[
                {'id':'Q1','query':'fixture query','knowledge_task_ids':['K1'],
                 'expected_information':'synthetic support','run_if':'always'}]},
                handoff_text='拟检索需求K1的资料；请结合工具实际返回的原文验证条件，不把计划当作检索结果。')
        else:
            doc=Books().doc
            result.update(decision='answer',final_answer={'answer':'A','evidence_status':'partial','summary':'fixture only'},
                citations=[{'evidence_id':doc['id'],'quote':doc['text'],'support':'partial'}],critical_gaps=['synthetic gap'])
        return SimpleNamespace(content=json.dumps(result,ensure_ascii=False),response_metadata={
            'model':'local-fixture','finish_reason':'stop','token_usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})


class CommunicationRuntimeTest(unittest.TestCase):
    def setUp(self):self.q=Question('q','Synthetic NOT condition, adult, 10 mg, time limit.',{k:k for k in 'ABCD'})

    def chat_run(self,mode):
        model=ScriptedChat();agents=ChatRoles(model,output_policy='decoupled',memory_selection='needs')
        books=Books();runtime=Runtime(agents,books,mode=mode,output_policy='decoupled',memory_policy='basic',memory_selection='needs',max_calls=4)
        return runtime.run(self.q),model,runtime,books

    def test_text_bypasses_acp_and_delivers_actual_model_authored_prose(self):
        with (patch('LangChain.medqa.runtime.ACPSession',side_effect=AssertionError('Text entered ACP')),
              patch('LangChain.medqa.g2cp_profile.message',side_effect=AssertionError('Text entered G2CP'))):
            result,model,runtime,books=self.chat_run('text')
        self.assertEqual(result['answer'],'A');self.assertEqual(len(model.messages),3)
        self.assertEqual(result['metrics']['acp_frames'],0)
        self.assertEqual(result['metrics']['natural_handoff_rendered'],0)
        prose=model.messages[1][1].content
        self.assertIn('请核实需求K1',prose)
        self.assertNotIn('key_conditions',prose);self.assertNotIn('knowledge_tasks',prose)
        self.assertIn('10 mg',prose);self.assertIn('NOT',prose)
        self.assertIn(books.doc['text'],model.messages[2][1].content)
        self.assertEqual(runtime.handshake()['communication']['protocol'],'natural_language')
        for e in result['events']:
            if 'model_input' in e:self.assertNotIn('control',e)

    def test_structured_dispatches_model_and_tool_operations_through_decoder(self):
        result,model,runtime,books=self.chat_run('structured')
        self.assertEqual(result['answer'],'A');self.assertEqual(len(model.messages),3)
        self.assertIn('key_conditions',model.messages[1][1].content)
        self.assertNotIn('_natural_handoffs',model.messages[1][1].content)
        frames=[unpack(bytes.fromhex(p)) for e in result['events'] if e.get('tool')=='acp_exchange' for p in e['frames_hex']]
        requests=[f for f in frames if f[0]==Kind.REQUEST]
        for opcode in (Opcode.PLAN,Opcode.QUERY_PLAN,Opcode.ANSWER,Opcode.SEARCH,Opcode.BATCH):
            self.assertTrue(any(f[3]==opcode for f in requests))
        self.assertEqual(result['metrics']['natural_handoff_chars'],0)
        self.assertGreater(result['metrics']['acp_object_registration_bytes'],0)
        self.assertGreater(result['metrics']['acp_evidence_ref_reads'],0)
        self.assertEqual(runtime.handshake()['communication']['ready'],[0,1,2,3,4])
        self.assertEqual(books.calls,1)
        self.assertEqual(result['metrics']['total_tokens'],360)
        for event,messages in zip([e for e in result['events'] if 'model_input' in e],model.messages):
            self.assertEqual(event['model_input'],messages[1].content)

    def test_conditional_batch_preserves_existing_executor_behavior(self):
        class Conditional(MinimalAgents):
            def call(self,role,p,mode):
                data,usage=super().call(role,p,mode)
                if role==INVESTIGATOR:
                    first=data['search']['queries'][0];first.update(id='Q1',run_if='always')
                    second={**first,'id':'Q2','query':'backup query','after':'Q1','run_if':'empty_or_duplicate'}
                    data['search']['queries']=[first,second]
                return data,usage
        for empty_until,expected in [(0,1),(1,2)]:
            for mode in ('text','structured'):
                books=Books(empty_until=empty_until)
                result=Runtime(Conditional(),books,mode=mode,output_policy='decoupled').run(self.q)
                self.assertEqual(books.calls,expected);self.assertEqual(result['answer'],'A')
                self.assertEqual(result['metrics']['model_calls'],3)

    def test_real_vector_consumer_and_no_extra_embedding_or_model_call(self):
        class Memory:
            def __init__(self):self.received=[]
            def recall(self,values,*args,**kwargs):self.received.append(values);return []
            def put(self,*args,**kwargs):raise AssertionError('writes disabled')
        class Encoder(Embeddings):
            def __init__(self):self.count=0
            def embed_text(self,text):self.count+=1;return super().embed_text(text)
        for mode in ('text','structured'):
            memory=Memory();encoder=Encoder()
            result=Runtime(MinimalAgents(),Books(),mode=mode,memory=memory,embedder=encoder,memory_write=False,
                output_policy='decoupled',memory_policy='basic',memory_selection='needs').run(self.q)
            self.assertEqual(len(memory.received),1);self.assertEqual(encoder.count,1)
            self.assertEqual(result['metrics']['model_calls'],3)
            self.assertEqual(result['metrics']['state_handle_reads'],int(mode=='structured'))
            self.assertEqual(list(memory.received[0]),[1.,0.,0.,0.])

    def test_ten_continuous_tasks_both_modes_no_new_call_or_budget_allowance(self):
        for mode in ('text','structured'):
            for index in range(10):
                q=Question('q'+str(index),self.q.question,self.q.options)
                r=Runtime(MinimalAgents(),Books(),mode=mode,output_policy='decoupled',max_calls=3).run(q)
                self.assertEqual(r['answer'],'A');self.assertEqual(r['metrics']['model_calls'],3)

    def test_no_json_document_disguised_as_natural_handoff(self):
        with self.assertRaises(ContractError):describe_output(PLANNER,{'handoff_text':'{"action":"search"}'})

    def test_report_keeps_protocol_cost_separate_from_model_usage(self):
        r,_,_,_=self.chat_run('structured');r['correct']=True
        m=summarize([r])
        self.assertEqual(m['total_tokens'],360)
        self.assertEqual(m['communication_versions'],['moreagent.handoff/1'])
        self.assertGreater(m['handoff_metrics']['acp_handshake_bytes'],0)


if __name__=='__main__':unittest.main()
