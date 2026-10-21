"""ACP-1: task-scoped binary requests, capabilities and shared object references.

This is an in-process transport/dispatch implementation. The bytes returned by
the codec are real messages; shared objects remain in the task's object store.
Registration bytes are counted separately, not presented as free compression.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import struct
import time

from .grounding import ContractError

VERSION = 1
COMMUNICATION_VERSION = 'moreagent.handoff/1'
HEADER = struct.Struct('!BBBBHIII')


class Actor(IntEnum):
    RUNTIME=0; PLANNER=1; INVESTIGATOR=2; REASONER=3; TEXTBOOKS=4; MEMORY=5


class Kind(IntEnum):
    REQUEST=1; RESULT=2; ERROR=3


class Opcode(IntEnum):
    HELLO=1; SEARCH=10; READ_SOURCE=11; READ_MEMORY=20; RECALL=21
    PUT_UNIT=22; PUT_DOCUMENT=23; ANSWER=40; PLAN=50; QUERY_PLAN=51; BATCH=60


class Status(IntEnum):
    OK=0; INVALID=1; FORBIDDEN=2; NOT_READY=3; TOOL_ERROR=4


class RunIf(IntEnum):
    ALWAYS=1; EMPTY_OR_DUPLICATE=2; TOOL_ERROR=3


class ACPError(ContractError):
    pass


@dataclass(frozen=True)
class Capability:
    owner: Actor
    opcode: Opcode
    parameters: tuple[str, ...]
    callers: tuple[Actor, ...]
    result: str = 'result_ref'


CAPABILITIES = (
    Capability(Actor.PLANNER,Opcode.PLAN,('task','context'),(Actor.RUNTIME,)),
    Capability(Actor.INVESTIGATOR,Opcode.QUERY_PLAN,('task','context'),(Actor.RUNTIME,)),
    Capability(Actor.REASONER,Opcode.ANSWER,('task','context'),(Actor.RUNTIME,)),
    Capability(Actor.TEXTBOOKS,Opcode.SEARCH,('query','top_k'),(Actor.RUNTIME,Actor.INVESTIGATOR)),
    Capability(Actor.TEXTBOOKS,Opcode.READ_SOURCE,('source',),(Actor.RUNTIME,)),
    Capability(Actor.MEMORY,Opcode.READ_MEMORY,('memory_ids',),(Actor.RUNTIME,)),
    Capability(Actor.MEMORY,Opcode.RECALL,('vector','recall_parameters'),(Actor.RUNTIME,)),
    Capability(Actor.MEMORY,Opcode.PUT_UNIT,('memory_unit','vector'),(Actor.RUNTIME,)),
    Capability(Actor.MEMORY,Opcode.PUT_DOCUMENT,('document','vector','metadata'),(Actor.RUNTIME,)),
    Capability(Actor.RUNTIME,Opcode.BATCH,('query_batch',),(Actor.INVESTIGATOR,)),
)


def pack(kind,src,dst,opcode,task,request,values):
    values=tuple(values)
    if any(type(v) is not int or not 0<=v<2**32 for v in values):
        raise ACPError('ACP参数必须为无符号32位整数')
    body=struct.pack('!'+'I'*len(values),*values)
    return HEADER.pack(VERSION,int(kind),int(src),int(dst),int(opcode),task,request,len(body))+body


def unpack(packet):
    if not isinstance(packet,bytes) or len(packet)<HEADER.size:
        raise ACPError('ACP消息头不完整')
    version,kind,src,dst,opcode,task,request,length=HEADER.unpack_from(packet)
    if version!=VERSION or not task or not request or length>65536 or length%4 or len(packet)!=HEADER.size+length:
        raise ACPError('ACP版本、标识或消息长度无效')
    try:
        result=(Kind(kind),Actor(src),Actor(dst),Opcode(opcode))
    except ValueError:
        raise ACPError('ACP消息类型、参与方或操作码未知') from None
    return (*result,task,request,struct.unpack('!'+'I'*(length//4),packet[HEADER.size:]))


class Session:
    def __init__(self, task_id, *, embedding_model='fixture', memory_enabled=True):
        if type(task_id) is not int or not 1<=task_id<2**32:
            raise ACPError('ACP任务ID无效')
        self.task_id=task_id
        self.embedding_model=embedding_model
        self.objects={}; self._dedup={}; self._next_ref=1; self._request=0
        self.ready=set(); self.handlers={}; self.frames=[]
        self.capabilities={(c.owner,c.opcode):c for c in CAPABILITIES if memory_enabled or c.owner!=Actor.MEMORY}
        self.metrics={k:0 for k in ('acp_request_bytes','acp_result_bytes','acp_handshake_bytes',
            'acp_object_registration_bytes','acp_frames','acp_operations','acp_batches',
            'acp_evidence_ref_reads','acp_memory_ref_reads','state_handle_reads','state_handle_bytes')}
        self.metrics.update(acp_dispatch_seconds=0.,acp_overhead_seconds=0.,acp_handler_seconds=0.)
        for actor in Actor:
            if actor!=Actor.MEMORY or memory_enabled:self.negotiate(actor)

    def put(self, kind, value):
        raw=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        key=(kind,hashlib.sha256(raw).digest())
        if key not in self._dedup:
            ident=self._next_ref;self._next_ref+=1
            self._dedup[key]=ident;self.objects[ident]=(kind,deepcopy(value))
            self.metrics['acp_object_registration_bytes']+=len(raw)
        return self._dedup[key]

    def get(self, ident, kind):
        item=self.objects.get(ident)
        if item is None or item[0]!=kind:
            raise ACPError('ACP引用不存在、类型不符或不属于本任务')
        return deepcopy(item[1])

    def prepare_context(self, payload):
        context={k:deepcopy(v) for k,v in payload.items() if k not in {'question_id','question','options'}}
        for name,kind in [('evidence_pool','evidence_view'),('memory_hits','memory_view'),
                          ('key_conditions','condition'),('knowledge_tasks','knowledge_task')]:
            if name in context:
                if not isinstance(context[name],list):raise ACPError('ACP角色上下文列表类型无效')
                context[name]={'references':[self.put(kind,item) for item in context[name]]}
        return context

    def resolve_context(self, context):
        context=deepcopy(context)
        for name,kind in [('evidence_pool','evidence_view'),('memory_hits','memory_view'),
                          ('key_conditions','condition'),('knowledge_tasks','knowledge_task')]:
            if name in context:
                envelope=context[name]
                if (not isinstance(envelope,dict) or set(envelope)!={'references'}
                        or not isinstance(envelope['references'],list)
                        or any(type(i) is not int for i in envelope['references'])):
                    raise ACPError('ACP角色上下文引用格式无效')
                context[name]=[self.get(i,kind) for i in envelope['references']]
                metric={'evidence_pool':'acp_evidence_ref_reads','memory_hits':'acp_memory_ref_reads'}.get(name)
                if metric:self.metrics[metric]+=len(context[name])
        return context

    def publish_vector(self, values):
        import numpy as np
        data=np.asarray(values,dtype='<f4')
        if data.ndim!=1 or not 1<=data.size<=8192 or not np.isfinite(data).all():
            raise ACPError('ACP向量维度或数值无效')
        raw=data.tobytes();key=('vector',hashlib.sha256(self.embedding_model.encode()+raw).digest())
        if key not in self._dedup:
            ident=self._next_ref;self._next_ref+=1;self._dedup[key]=ident
            self.objects[ident]=('vector',{'raw':raw,'model':self.embedding_model,'dim':data.size})
            self.metrics['acp_object_registration_bytes']+=len(raw)
        return self._dedup[key]

    def consume_vector(self, ident):
        import numpy as np
        item=self.objects.get(ident)
        if item is None or item[0]!='vector' or item[1]['model']!=self.embedding_model:
            raise ACPError('ACP向量句柄或编码模型不匹配')
        data=item[1]
        if len(data['raw'])!=data['dim']*4:
            raise ACPError('ACP向量数据长度不符')
        self.metrics['state_handle_reads']+=1
        self.metrics['state_handle_bytes']+=len(data['raw'])
        return np.frombuffer(data['raw'],dtype='<f4')

    def negotiate(self, actor, versions=(VERSION,)):
        if VERSION not in versions:
            raise ACPError('ACP没有共同支持的版本')
        descriptors=[{'opcode':int(c.opcode),'parameters':c.parameters,'result':c.result}
                     for c in self.capabilities.values() if c.owner==actor]
        reference=self.put('capabilities',descriptors)
        self._request+=1
        request=pack(Kind.REQUEST,actor,Actor.RUNTIME,Opcode.HELLO,self.task_id,self._request,(VERSION,reference))
        k,src,dst,op,task,req,args=unpack(request)
        if args!=(VERSION,reference) or self.get(args[1],'capabilities')!=descriptors:
            raise ACPError('ACP能力协商失败')
        ack=pack(Kind.RESULT,Actor.RUNTIME,actor,Opcode.HELLO,task,req,(int(Status.OK),VERSION))
        unpack(ack)
        self.frames.extend([request,ack]);self.metrics['acp_handshake_bytes']+=len(request)+len(ack)
        self.metrics['acp_frames']+=2;self.ready.add(actor)

    def bind(self, actor, opcode, handler):
        key=(Actor(actor),Opcode(opcode))
        if key not in self.capabilities or not callable(handler):
            raise ACPError('ACP只能绑定已声明的能力')
        self.handlers[key]=handler

    def _arguments(self, cap, values):
        if len(values)!=len(cap.parameters):
            raise ACPError('ACP位置参数数量错误')
        result=[]
        for name,value in zip(cap.parameters,values):
            if name=='top_k':
                if not 1<=value<=100:raise ACPError('ACP top_k超出范围')
                result.append(value)
            elif name=='vector':
                # Validate every argument before consuming any vector or executing.
                if value not in self.objects or self.objects[value][0]!='vector':
                    raise ACPError('ACP向量句柄不存在')
                result.append(value)
            else:
                item=self.get(value,name)
                if name in {'task','context','recall_parameters','memory_unit','metadata','document','query_batch'} and not isinstance(item,dict):
                    raise ACPError('ACP对象参数类型错误')
                if name in {'query','source'} and (not isinstance(item,str) or not item.strip() or len(item)>2000):
                    raise ACPError('ACP查询或来源参数无效')
                if name=='memory_ids' and (not isinstance(item,list) or not 1<=len(item)<=10 or
                        any(not isinstance(i,str) or not i for i in item) or len(set(item))!=len(item)):
                    raise ACPError('ACP记忆引用参数无效')
                result.append(item)
        return [self.consume_vector(v) if name=='vector' else v for name,v in zip(cap.parameters,result)]

    def receive(self, packet):
        kind,src,dst,opcode,task,request,values=unpack(packet)
        if kind!=Kind.REQUEST or task!=self.task_id:
            raise ACPError('ACP消息类型或任务不匹配')
        if src not in self.ready or dst not in self.ready:
            raise ACPError('ACP参与方尚未完成协商')
        cap=self.capabilities.get((dst,opcode))
        if cap is None or src not in cap.callers or (dst,opcode) not in self.handlers:
            raise ACPError('ACP能力未注册或调用方无权限')
        parameters=self._arguments(cap,values)
        started=time.perf_counter()
        try:
            result=self.handlers[(dst,opcode)](*parameters)
        finally:
            self.metrics['acp_handler_seconds']+=time.perf_counter()-started
        ref=self.put('result',result)
        return pack(Kind.RESULT,dst,src,opcode,task,request,(int(Status.OK),ref))

    def call(self, src, dst, opcode, values):
        started=time.perf_counter();self._request+=1
        before_handler=self.metrics['acp_handler_seconds']
        request_id=self._request
        packet=pack(Kind.REQUEST,src,dst,opcode,self.task_id,request_id,values)
        self.frames.append(packet);self.metrics['acp_frames']+=1
        self.metrics['acp_request_bytes']+=len(packet)
        try:
            reply=self.receive(packet)
            kind,r_src,r_dst,r_op,task,request,args=unpack(reply)
            if (kind,r_src,r_dst,r_op,task,request)!=(Kind.RESULT,Actor(dst),Actor(src),Opcode(opcode),self.task_id,request_id):
                raise ACPError('ACP响应关联错误')
            if len(args)!=2 or args[0]!=Status.OK:raise ACPError('ACP结果状态错误')
            self.frames.append(reply);self.metrics['acp_frames']+=1
            self.metrics['acp_result_bytes']+=len(reply);self.metrics['acp_operations']+=1
            return self.get(args[1],'result')
        except Exception as error:
            reference=self.put('error',{'error_type':type(error).__name__})
            status=Status.INVALID if isinstance(error,ACPError) else Status.TOOL_ERROR
            reply=pack(Kind.ERROR,dst,src,opcode,self.task_id,request_id,(int(status),reference))
            self.frames.append(reply);self.metrics['acp_frames']+=1
            self.metrics['acp_result_bytes']+=len(reply)
            raise
        finally:
            elapsed=time.perf_counter()-started
            self.metrics['acp_dispatch_seconds']+=elapsed
            self.metrics['acp_overhead_seconds']+=max(0.,elapsed-(self.metrics['acp_handler_seconds']-before_handler))

    def map_tool_plan(self, output):
        """Compile typed batch references, then validate them on receipt.

        The existing search executor still determines conditions, budgets and
        actual tool order. This path does not add model calls or parallelism.
        """
        search=output.get('search')
        if not isinstance(search,dict) or not isinstance(search.get('queries'),list) or not 1<=len(search['queries'])<=12:
            raise ACPError('ACP批次需要1至12条查询')
        ids={};items=[]
        for index,query in enumerate(search['queries'],1):
            if not isinstance(query,dict):raise ACPError('ACP查询参数应为对象')
            ident=query.get('id','Q'+str(index))
            if not isinstance(ident,str) or ident in ids:raise ACPError('ACP查询ID无效或重复')
            condition=query.get('run_if','always')
            table={'always':RunIf.ALWAYS,'empty_or_duplicate':RunIf.EMPTY_OR_DUPLICATE,'tool_error':RunIf.TOOL_ERROR}
            if condition not in table:raise ACPError('ACP查询执行条件未知')
            after=ids.get(query.get('after'),0)
            if condition!='always' and not after:raise ACPError('ACP条件操作必须依赖前面的操作')
            if (not isinstance(query.get('query'),str) or not query['query'].strip()
                    or not isinstance(query.get('expected_information'),str) or not query['expected_information'].strip()
                    or not isinstance(query.get('knowledge_task_ids'),list) or not query['knowledge_task_ids']):
                raise ACPError('ACP查询缺少文本、需求关联或预期信息')
            ids[ident]=index
            items.append([index,int(Actor.TEXTBOOKS),int(Opcode.SEARCH),self.put('query_parameters',query),after,int(table[condition])])
        base=deepcopy(output);base['search'].pop('queries')
        reference=self.put('query_batch',{'items':items,'base':base})
        return self.call(Actor.INVESTIGATOR,Actor.RUNTIME,Opcode.BATCH,(reference,))

    def accept_batch(self, batch):
        seen=set();labels={};queries=[]
        for op_id,dst,opcode,parameters,after,condition in batch['items']:
            if op_id in seen or dst!=Actor.TEXTBOOKS or opcode!=Opcode.SEARCH or after and after not in seen:
                raise ACPError('ACP批次操作、目标或依赖无效')
            try: rule=RunIf(condition)
            except ValueError:raise ACPError('ACP批次条件未知') from None
            if rule!=RunIf.ALWAYS and not after:raise ACPError('ACP条件操作缺少前驱')
            query=self.get(parameters,'query_parameters')
            query['run_if']={RunIf.ALWAYS:'always',RunIf.EMPTY_OR_DUPLICATE:'empty_or_duplicate',RunIf.TOOL_ERROR:'tool_error'}[rule]
            if rule!=RunIf.ALWAYS:query['after']=labels[after]
            labels[op_id]=query.get('id','Q'+str(op_id));queries.append(query);seen.add(op_id)
        self.metrics['acp_batches']+=1
        result=deepcopy(batch['base']);result['search']['queries']=queries
        return result

    def description(self):
        return {'protocol':'ACP-1','version':VERSION,'transport':'in_process_binary_dispatch',
            'ready':[int(a) for a in sorted(self.ready)],
            'capabilities':[{'agent':int(c.owner),'opcode':int(c.opcode),'parameters':list(c.parameters),
                             'result':c.result,'callers':[int(a) for a in c.callers]} for c in self.capabilities.values()],
            'state_format':'task_scoped_float32_handle','object_registration_counted':True}
