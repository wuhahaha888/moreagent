"""Bounded memory previews and deterministic, need-aware selection.

Coverage is the Planner's *proposed* need mapping, never medical verification.
No additional model/embedding calls are made by this module.
"""
from copy import deepcopy
import re

VERSION = 'needs-memory-v3'
PLANNER_LIMIT = 3
PLANNER_PREVIEW_CHARS = 1000
STAGES = ('recalled','presented','planner_accepted','selected','resolved','delivered','executed','cited')
KINDS = ('evidence','strategy','episode')


def kind(hit):
    return 'evidence' if hit.get('kind','textbook_extract')=='textbook_extract' else hit['kind']


def previews(hits, max_chars, serialize):
    """Only prefiltered headers enter a model; never expose all coarse hits."""
    result=[]
    for hit in hits[:PLANNER_LIMIT]:
        item={'id':hit['id'],'kind':kind(hit),'summary':hit['summary'][:160]}
        # These describe the previous case, not a general applicability guarantee.
        conditions=[c for c in hit.get('conditions',[]) if isinstance(c,str)]
        if conditions:item['origin_conditions']=[c[:80] for c in conditions[:2]]
        if len(serialize([*result,item]))>min(max_chars,PLANNER_PREVIEW_CHARS):
            # Drop a lower-priority candidate rather than shrinking all summaries
            # into opaque fragments or deleting only their limiting conditions.
            continue
        result.append(item)
    return result


def result_headers(hits):
    return [{'id':h['id'],'kind':kind(h),'summary':h['summary'][:20]} for h in hits]


def intent(text):
    """Weak intent labels. Treatment mechanisms remain eligible complements."""
    patterns=(
        ('treatment',r'\b(treat\w*|therap\w*|management|antidote|indications?)\b|uses of|next step|治疗|解毒|用药|适应证|处置'),
        ('diagnosis',r'\b(diagnos\w*|etiolog\w*)\b|most likely cause|诊断|病因|致病'),
        ('mechanism',r'\b(mechanis\w*|receptors?|physiolog\w*)\b|increased by|机制|受体|分泌|药理'),
        ('research',r'\b(study design|odds ratio|relative risk|bias|confidence interval)\b|研究设计|偏倚|置信区间'),
    )
    for label,pattern in patterns:
        if re.search(pattern,text,re.I):return label
    return 'unknown'


def task_match(question, hit, task_goals=()):
    current=intent(question)
    if current=='unknown':current=intent(' '.join(task_goals))
    historical=hit.get('content',{}).get('task_intent') or intent(hit.get('task_topic',''))
    if historical=='unknown':
        content=hit.get('content',{})
        historical=intent(' '.join(content.get('needs',[]))+ ' '+hit.get('summary',''))
    if current=='unknown' or historical=='unknown':return .5
    if current==historical:return 1.
    if {current,historical} <= {'treatment','diagnosis','mechanism'}:return .35
    return .1


def words(text):
    return set(re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]{2,}',text.casefold()))


def applicability(question, hit):
    """Literal metadata checks only, never infer clinical applicability.

    Historical patient conditions are deliberately not mandatory constraints.
    Existing records usually have no explicit applicability rule and stay unknown.
    """
    rules=hit.get('content',{}).get('applicability')
    if not rules:return .5,'requires_current_task_review',False
    if not isinstance(rules,dict):return 0.,'invalid_rule_metadata',True
    required,excluded=rules.get('required_terms',[]),rules.get('excluded_terms',[])
    if any(not isinstance(items,list) or any(not isinstance(t,str) or not t.strip() for t in items)
           for items in (required,excluded)):
        return 0.,'invalid_rule_metadata',True
    def positive(term):
        pattern=(r'(?<!\w)'+re.escape(term)+r'(?!\w)' if re.fullmatch(r'[\w -]+',term,re.ASCII)
                 else re.escape(term))
        for match in re.finditer(pattern,question,re.I):
            before=question[max(0,match.start()-35):match.start()]
            # Ambiguous negated mentions cannot establish an exclusion.
            if not re.search(r'\b(no|not|without|denies|negative for)\b|无|非|未|否认',before,re.I):
                return True
        return False
    if any(positive(t) for t in excluded):return 0.,'explicit_excluded_term_present',True
    if required and not all(positive(t) for t in required):
        return .25,'required_terms_not_established',False
    return (.8,'explicit_terms_match_requires_review',False) if required else (.5,'requires_current_task_review',False)


def prefilter(question, hits, limit=PLANNER_LIMIT):
    """Program-only coarse ranking before Planner, with auditable uncertainty.

    R is cosine relevance; T is a weak intent rule; A is explicit metadata or
    unknown. No source-task patient conditions are promoted to universal rules.
    """
    ranked,records=[],[]
    for ident,hit in hits.items():
        relevance=max(0.,min(1.,float(hit.get('similarity') or 0)))
        match=task_match(question,hit)
        fit,status,blocked=applicability(question,hit)
        score=.45*relevance+.35*match+.20*fit
        record={'memory_id':ident,'R':relevance,'T':match,'A':fit,'score':score,
                'applicability_status':status,'selected':False,
                'reason':'explicit_condition_or_metadata' if blocked else 'below_prefilter_floor' if score<.5 else 'rank_or_budget'}
        records.append(record)
        if not blocked and score>=.5:ranked.append((score,relevance,ident,hit,record))
    selected=[]
    for _,_,ident,hit,record in sorted(ranked,key=lambda x:(-x[0],-x[1],x[2])):
        duplicate=False
        for old in selected:
            other=hits[old]
            if kind(other)!=kind(hit):continue
            a,b=hit.get('content',{}),other.get('content',{})
            same_source=(kind(hit)=='evidence' and a.get('sha256') and a.get('sha256')==b.get('sha256'))
            left,right=words(hit.get('summary','')),words(other.get('summary',''))
            overlap=len(left&right)/max(1,len(left|right))
            if same_source or (left and overlap>=.75):duplicate=True;break
        if duplicate:record['reason']='duplicate_source_or_high_lexical_overlap';continue
        if len(selected)>=min(limit,PLANNER_LIMIT):continue
        selected.append(ident);record.update(selected=True,reason='prefiltered_for_planner')
    return selected,records


def select(question, hits, assessments, need_ids, limit=3, *, task_goals=()):
    """Greedy 0.45R+0.35T+0.20C with marginal coverage and redundancy penalty."""
    remaining={a['memory_id']:a for a in assessments if a['decision']=='read' and a['memory_id'] in hits}
    selected,records=[],[]
    covered={k:set() for k in KINDS}
    while remaining and len(selected)<limit:
        ranked=[]
        for ident,assessment in remaining.items():
            hit=hits[ident]
            relevance=max(0.,min(1.,float(hit.get('similarity') or 0)))
            match=task_match(question,hit,task_goals)
            mapped=set(assessment['knowledge_task_ids']) & set(need_ids)
            # A query hint does not consume the coverage credit of source evidence.
            coverage=len(mapped-covered[kind(hit)])/max(1,len(need_ids))
            terms=words(hit.get('summary',''))
            redundancy=0.
            for old in selected:
                other=hits[old]
                if kind(other)!=kind(hit):continue
                other_terms=words(other.get('summary',''))
                overlap=len(terms & other_terms)/max(1,len(terms | other_terms))
                if overlap>=.75:redundancy=max(redundancy,.15)
                a,b=hit.get('content',{}),other.get('content',{})
                if kind(hit)=='evidence' and a.get('sha256') and a.get('sha256')==b.get('sha256'):
                    redundancy=1.
            score=.45*relevance+.35*match+.20*coverage-redundancy
            ranked.append((score,relevance,ident,{'memory_id':ident,'kind':kind(hit),'R':relevance,'T':match,
                'C':coverage,'redundancy_penalty':redundancy,'score':score,'need_ids':sorted(mapped),
                'coverage_status':'planner_proposed_not_verified'}))
        score,_,ident,record=sorted(ranked,key=lambda x:(-x[0],-x[1],x[2]))[0]
        remaining.pop(ident)
        if record['redundancy_penalty']>=1:continue
        selected.append(ident);records.append(record);covered[record['kind']].update(record['need_ids'])
    return selected,records


class ReuseTrace:
    def __init__(self):
        self.ids={stage:{k:set() for k in KINDS} for stage in STAGES}

    def mark(self,stage,ident,hit):
        self.ids[stage][kind(hit)].add(ident)

    def export(self):
        return {stage:{k:len(ids) for k,ids in kinds.items()} for stage,kinds in self.ids.items()}

    def references(self):
        return {stage:{k:sorted(ids) for k,ids in kinds.items()} for stage,kinds in self.ids.items()}
