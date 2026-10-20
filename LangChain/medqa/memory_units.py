"""Three small, typed memory units, built from observations without LLM calls.

Evidence stores exact source excerpts, not a verified model diagnosis. Strategy
and episode records describe executions and are never medical evidence.
"""
from copy import deepcopy
import hashlib
from .memory_selection import intent

KINDS = ('evidence', 'strategy', 'episode')
POLICY_VERSION = 'basic-memory-units-v1'


def normalize_unit(unit, trace):
    unit = deepcopy(unit)
    if unit.get('kind') not in KINDS:
        raise ValueError('未知记忆类型')
    for key in ('source_agent', 'topic', 'summary'):
        if not isinstance(unit.get(key), str) or not unit[key].strip():
            raise ValueError('记忆元数据缺少 '+key)
    if len(unit['summary']) > 600:
        raise ValueError('记忆摘要超过600字符')
    content = unit.get('content')
    if not isinstance(content, dict) or any(not isinstance(content.get(k), str) or not content[k]
            for k in ('corpus_version', 'task_group', 'origin_question_id')):
        raise ValueError('记忆来源/作用域无效')
    content.update(schema_version=1, memory_type=unit['kind'], review_policy=POLICY_VERSION,created_by='runtime')
    if unit['kind'] == 'evidence':
        doc = unit.pop('document', None)
        if (not isinstance(doc, dict) or not isinstance(doc.get('text'), str)
                or hashlib.sha256(doc['text'].encode()).hexdigest() != doc.get('sha256')
                or any(content.get(k) != doc.get(k) for k in ('id','sha256','corpus_version'))):
            raise ValueError('事实记忆来源校验失败')
        excerpts = content.get('excerpts')
        if (not isinstance(excerpts, list) or not 1 <= len(excerpts) <= 3
                or any(not isinstance(q,str) or not q.strip() or q not in doc['text'] for q in excerpts)):
            raise ValueError('事实记忆必须为真实原文片段')
        content['validation'] = 'source_excerpt_checked; applicability_requires_current_task_review'
    else:
        if 'document' in unit:
            raise ValueError('过程记忆不能伪装成教材原文')
        content['validation'] = 'execution_observation_only; not_medical_evidence'
        # Stable on outbox retry, distinct across separate executions.
        content['origin_trace'] = trace
    # Runtime never receives grading labels. Reject accidental grading payloads.
    if any(k in content for k in ('correct','gold','gold_answer','reference_answer','answer_idx')):
        raise ValueError('记忆不能包含评分答案')
    return unit


def embedding_text(unit):
    content = unit['content']
    parts = [content['task_group'], unit['kind'], unit['topic'], unit['summary']]
    parts.extend(content.get('conditions', []))
    if unit['kind'] == 'evidence':
        parts.extend([content.get('title',''), *content.get('excerpts',[])])
    elif unit['kind'] == 'strategy':
        parts.extend(q['query'] for q in content.get('queries',[]))
    return '\n'.join(parts)


def build_units(question, state, result, evidence, history, metrics, *, actionable_strategy=False):
    """At most two source units + one strategy + one episode, no new model turn."""
    topic = question.question[:240]
    common = {'corpus_version': state['corpus_version'], 'task_group': state['task_group'],
              'origin_question_id': question.id,
              'task_intent':intent(question.question),
              'needs':[t['goal'] for t in state.get('knowledge_tasks',[])],
              'conditions': [c['text'] for c in state.get('key_conditions',[])][:12],
              'conditions_role': 'source_task_context; not_a_general_applicability_guarantee'}
    units, grouped = [], {}
    for c in result.get('evidence_result',{}).get('citations',[]):
        ident, quote = c.get('evidence_id'), c.get('quote')
        if (c.get('source_verified') and c.get('visible') and ident in evidence
                and c.get('model_reported_support') in {'supported','partial','contradicted'}
                and isinstance(quote,str) and quote in evidence[ident]['text']):
            grouped.setdefault(ident, []).append(c)
    for ident, citations in list(grouped.items())[:2]:
        doc = evidence[ident]
        excerpts = list(dict.fromkeys(c['quote'] for c in citations))[:3]
        content = {**deepcopy(common), **{k:doc[k] for k in ('id','title','sha256')},
                   'excerpts': excerpts, 'model_reported_support': citations[0]['model_reported_support']}
        units.append({'kind':'evidence','source_agent':'VerifierReasoner','topic':topic,
                      'summary':'\n'.join(excerpts)[:600], 'document':deepcopy(doc),'content':content})
    queries = []
    for observation in history[-6:]:
        if not observation.get('query'):
            continue
        ids = observation.get('evidence_ids',[])
        queries.append({'query':observation['query'], 'source_ids':ids,
                        'outcome':'error' if observation.get('error') else 'candidates' if ids else 'empty',
                        'new_source_count':len(observation.get('new_sources',[])),
                        'cited_source_ids':[i for i in ids if i in grouped]})
    if queries:
        # Surface one concrete reusable query, not only an opaque execution count.
        # Citation counts are observations, never proof of clinical correctness.
        if actionable_strategy:queries.sort(key=lambda q:(-len(q['cited_source_ids']),-q['new_source_count']))
        summary=(('查询线索：'+queries[0]['query'][:130]+'；本次'+str(len(queries))+'个查询，'+
                   str(sum(bool(q['source_ids']) for q in queries))+'个返回候选（仅执行观测）')[:240]
                 if actionable_strategy else ('本次检索执行记录：'+str(len(queries))+'个查询，'+
                   str(sum(bool(q['source_ids']) for q in queries))+'个返回候选；是否被引用只表示使用观测，不证明查询或医学结论正确。')[:600])
        units.append({'kind':'strategy','source_agent':'Investigator','topic':topic,
            'summary':summary,
            'content':{**deepcopy(common),'queries':queries,
                       'source_ids':sorted({i for q in queries for i in q['source_ids']}),
                       'needs':[t['goal'] for t in state.get('knowledge_tasks',[])],
                       'reuse_rule':'只作查询策略提示；不能把过去空结果当作当前必然无证据。'}})
    gaps = [g for g in result.get('gaps',state.get('gaps',[])) if isinstance(g,str)][:3]
    needs=[t['goal'] for t in state.get('knowledge_tasks',[])]
    units.append({'kind':'episode','source_agent':'Planner','topic':topic,
        'summary':('需求：'+'；'.join(n[:60] for n in needs[:2])+ '；未解决：'+
                   '；'.join(g[:60] for g in gaps[:2])+'；状态：'+str(result.get('status','unknown'))+
                   '；检索'+str(metrics['retrieval_count'])+'次')[:240],
        'content':{**deepcopy(common),'needs':[t['goal'] for t in state.get('knowledge_tasks',[])],
                   'source_ids':sorted(evidence),'outcome':result.get('status','unknown'),
                   'gaps':gaps, 'tool_counts':{'searches':metrics['retrieval_count'],'source_reads':metrics['document_reads']},
                   'conclusion_status':'No medical conclusion or answer letter promoted from this record.'}})
    return units
