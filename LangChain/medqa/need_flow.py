"""Need-first memory selection and validated control decisions, without LLM calls."""
from copy import deepcopy

from .grounding import ContractError, identifiers
from .task_state import TaskStates
from .evidence_processor import SEPARATOR, sha, quote_ranges

VERSION = 'moreagent.need-first/4'
FRAGMENT_LIMIT = 9  # Independent of the <=3 unique-source read budget; IPC resolves <=10 IDs.


def fragment_citations(fragments, accepted_ids, result, evidence, reviews):
    """Attribute actual saved spans, never merely another quote from the source.

    This measures citation use, not independent correctness or causal benefit.
    """
    valid = {(c['evidence_id'],c['quote']) for c in result.get('evidence_result',{}).get('citations',[])
             if c.get('source_verified') and c.get('visible')}
    used, supported = set(), set()
    for ident in accepted_ids:
        fragment = fragments.get(ident,{})
        source = evidence.get(fragment.get('source_id'),{})
        if not source or source.get('sha256') != fragment.get('source_sha256'):
            continue
        matching = set()
        for source_id, quote in valid:
            if source_id != fragment['source_id'] or not quote:
                continue
            start = source['text'].find(quote)
            while start >= 0:
                end = start + len(quote)
                if any((s['start'] <= start and end <= s['end'])
                       or (start <= s['start'] and s['end'] <= end)
                       for s in fragment.get('spans',[]) if s['end'] > s['start']):
                    matching.add((source_id,quote));break
                start = source['text'].find(quote,start+1)
        if matching:
            used.add(ident)
        for review in reviews:
            if (review.get('verdict') == 'supported' and review.get('condition_verdict') == 'matched'
                    and any((q.get('evidence_id'),q.get('quote')) in matching
                            for q in review.get('support_quotes',[]))):
                supported.add(review['task_id'])
    return used, supported


def need_query(task, conditions):
    related = set(task.get('condition_ids', []))
    return '\n'.join([task['goal'], *[c['text'] for c in conditions if c['id'] in related]])


def source_key(hit):
    content = hit['content']
    return content['id'], content['sha256']


def select_sources(rankings, candidates, limit, *, known_sources=(), fragment_limit=FRAGMENT_LIMIT):
    """Round-robin needs: retain memory IDs, charge only distinct new sources.

    Similarity chooses what to read; it never certifies coverage/applicability.
    """
    if fragment_limit <= 0:
        return []
    selected, seen_sources = [], set(known_sources)
    new_sources = set()
    lists = [list(ids) for ids in rankings.values()]
    for position in range(max(map(len, lists), default=0)):
        for ids in lists:
            if position >= len(ids):
                continue
            ident = ids[position]
            source = source_key(candidates[ident])
            if ident in selected:
                continue
            if source not in seen_sources:
                if len(new_sources) >= max(0,limit):
                    continue
                new_sources.add(source); seen_sources.add(source)
            selected.append(ident)
            if len(selected) >= fragment_limit:
                return selected
    return selected


def investigation_reviews(updates, tasks, evidence, visible, *, memory_sources=None):
    if (not isinstance(updates, list)
            or any(not isinstance(u,dict) or not isinstance(u.get('need_id'),str) for u in updates)
            or {u['need_id'] for u in updates} != set(tasks)):
        raise ContractError('调查阶段必须逐项提交当前需求的覆盖判断')
    seen, result = set(), {}
    for item in updates:
        if not isinstance(item, dict) or item.get('need_id') in seen:
            raise ContractError('调查需求状态无效或重复')
        ident = item['need_id']; seen.add(ident)
        status = item.get('coverage')
        if status not in {'covered', 'partial', 'missing', 'conflicted', 'unusable'}:
            raise ContractError('未知证据覆盖状态')
        raw_ids = item.get('evidence_ids', [])
        if not isinstance(raw_ids,list) or any(not isinstance(i,str) for i in raw_ids):
            raise ContractError('覆盖判断的来源引用须为ID列表')
        # A resolved memory is an unambiguous alias for its validated source.
        # Unknown/unread memories and invisible sources remain invalid.
        aliases = memory_sources or {}
        canonical = [aliases.get(i,i) if i not in evidence else i for i in raw_ids]
        ids = identifiers(list(dict.fromkeys(canonical)), set(visible) & set(evidence), empty=True)
        condition = item.get('condition_verdict', 'unknown')
        if condition not in {'matched', 'mismatched', 'unknown'}:
            raise ContractError('证据适用条件状态无效')
        if status in {'covered','partial','conflicted'} and not ids:
            raise ContractError('覆盖判断必须关联本次可见证据')
        if status == 'covered' and condition != 'matched':
            raise ContractError('暂时覆盖须明确当前条件匹配')
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ContractError('覆盖判断需要简短理由')
        result[ident] = {'coverage':status, 'coverage_evidence_ids':sorted(ids),
                         'condition_verdict':condition, 'coverage_reason':item['reason']}
    return result


def apply_investigation_reviews(states, reviews):
    for ident, values in reviews.items():
        # Provisional coverage does not turn a model judgment into verified fact.
        lifecycle = 'completed' if states.items[ident]['support']=='supported' else 'waiting_review'
        states.change(ident, lifecycle, 'investigator_coverage', **deepcopy(values))


def reasoner_reviews(output, states, evidence, visible):
    """Resolve citation indices locally; validate the whole update atomically."""
    reviews = output.get('need_reviews')
    if (not isinstance(reviews, list)
            or any(not isinstance(r,dict) or not isinstance(r.get('task_id'),str) for r in reviews)
            or {r['task_id'] for r in reviews} != set(states.items)):
        raise ContractError('核验阶段必须逐项更新需求状态')
    citations = output.get('citations', [])
    if not isinstance(citations, list):
        raise ContractError('需求引用须对应实际citations列表')
    expanded = []
    for review in reviews:
        if not isinstance(review, dict):
            raise ContractError('需求核验项须为对象')
        value = deepcopy(review)
        if value.get('condition_verdict') not in {'matched','mismatched','unknown'}:
            raise ContractError('需求核验的条件状态无效')
        indexes = value.pop('citation_indices', [])
        if (not isinstance(indexes, list)
                or any(type(i) is not int or not 0 <= i < len(citations) for i in indexes)
                or len(indexes) != len(set(indexes))):
            raise ContractError('需求引用序号不存在或重复')
        # The model supplies indices only. It cannot add unseen alternate quotes.
        value['support_quotes'] = [deepcopy(citations[i]) for i in indexes]
        expanded.append(value)
    working = TaskStates([], resource_snapshot=states.resource_snapshot)
    working.items = deepcopy(states.items)
    working.review_needs(expanded, evidence, visible)
    for review in expanded:
        item = working.items[review['task_id']]
        working.change(review['task_id'], item['lifecycle'], 'need_applicability_review',
                       condition_verdict=review['condition_verdict'])
        if review['verdict'] != 'supported':
            # A rejected provisional cover must not suppress the next search.
            working.change(review['task_id'], 'waiting_evidence', 'coverage_reopened_by_reasoner',
                           coverage={'partial':'partial','contradicted':'conflicted'}.get(review['verdict'],'missing'))
    request_investigation(output, working.items)
    states.items = working.items
    states.transitions.extend(working.transitions)
    return expanded


def request_investigation(output, tasks):
    if output.get('next_action') not in {'finish','investigate'}:
        raise ContractError('核验阶段须明确next_action=finish或investigate')
    if output['next_action'] == 'finish':
        return None
    request = output.get('investigation_request')
    if not isinstance(request, dict) or request.get('may_change_answer') is not True:
        raise ContractError('返回调查须声明可能影响结论的具体缺口')
    ids = identifiers(request.get('need_ids'), tasks)
    if any(tasks[i]['support'] == 'supported' for i in ids):
        raise ContractError('已支持需求须先通过新核验明确重新打开')
    for key in ('gap','expected_information'):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ContractError('返回调查需要缺口与预期信息')
    return {**deepcopy(request), 'need_ids':sorted(ids)}


def need_view(states):
    return {ident:{k:deepcopy(value[k]) for k in ('support','evidence_ids','coverage','coverage_evidence_ids','condition_verdict') if k in value}
            for ident,value in states.items.items()}


def compact_memory_view(pack, documents, excerpts):
    """Keep original offsets, exact saved quotes and local context; never summarize.

    Freshly searched sources must be excluded by the caller. Only shorten a view;
    the original document and ordinary expansion/retrieval remain available.
    """
    result = deepcopy(pack)
    for view in result['views']:
        ident = view['id']; doc = documents[ident]; text = doc['text']
        merged = quote_ranges(text,excerpts.get(ident,[]))
        rendered = SEPARATOR.join(text[a:b] for a,b in merged)
        if not merged or len(rendered) >= len(view['text']):
            continue
        spans = [{'id':ident+':'+sha(doc['sha256']+':'+str(a)+':'+str(b))[:12],
                  'start':a,'end':b,'text':text[a:b]} for a,b in merged]
        result['visible'][ident] = spans
        view['text'] = rendered
        view['has_more'] = merged != [(0,len(text))]
        result['records'].append({'source_id':ident,'action':'memory_excerpt_with_context',
                                  'ranges':merged,'raw_sha256':doc['sha256']})
    result['view_chars'] = sum(len(d['text']) for d in result['views'])
    return result
