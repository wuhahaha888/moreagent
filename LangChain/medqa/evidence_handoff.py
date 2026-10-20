"""Budgeted follow-up delivery, without a judge call or a fixed old/new ratio.

Only verbatim source intervals are selected. Citation/conflict labels come from
the existing response; lexical scores order candidates, never certify support.
The normal EvidenceProcessor algorithm and its on/off switch are unchanged.
"""
from copy import deepcopy
import math
import time

from .evidence_processor import SEPARATOR, sha, terms, validate_view

VERSION = 'followup-handoff-v1'


def merge_ranges(ranges):
    merged = []
    for a, b in sorted(ranges):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def subtract_ranges(ranges, seen):
    remaining = list(ranges)
    for left, right in merge_ranges(seen):
        pieces = []
        for a, b in remaining:
            if right <= a or left >= b:
                pieces.append((a, b))
            else:
                pieces.extend((x, y) for x, y in ((a, min(b, left)), (max(a, right), b)) if y > x)
        remaining = pieces
    return remaining


def visible_delta(pack, previous, documents):
    """Count newly delivered source text, not new IDs or newly stored raw files."""
    seen_by_hash, seen_text = {}, []
    for ident, spans in previous.get('visible', {}).items():
        if ident in documents:
            seen_by_hash.setdefault(documents[ident]['sha256'], []).extend(
                (s['start'], s['end']) for s in spans)
            seen_text.extend(s['text'] for s in spans)
    result, counted = {}, set()
    for view in pack.get('views', []):
        ident = view['id']
        doc = documents[ident]
        fresh = subtract_ranges([(s['start'], s['end']) for s in pack['visible'][ident]],
                                seen_by_hash.get(doc['sha256'], []))
        count = 0
        for a, b in fresh:
            text = doc['text'][a:b]
            # A different source/offset carrying the identical excerpt is no progress.
            if not text.strip() or any(text in old for old in seen_text):
                continue
            key = (doc['sha256'], a, b)
            if key not in counted:
                count += len(text)
                counted.add(key)
        if count:
            result[ident] = count
    return {'evidence_ids': sorted(result), 'characters': sum(result.values())}


def build_handoff(processor, documents, *, previous, citations, request, question,
                  targets=None, conflicts=()):
    started = time.monotonic()
    targets = targets or {}
    validate_view(previous, documents)
    for doc in documents.values():
        if sha(doc['text']) != doc['sha256']:
            raise ValueError('补查证据原文校验失败')
    # Preserve the user's question separately; prioritize the gap and proposed
    # English queries in extraction even when the gap itself is written in Chinese.
    queries = [q if isinstance(q, str) else q.get('query', '')
               for q in request.get('queries', []) if isinstance(q, (str, dict))]
    focus = '\n'.join([request.get('gap', ''), request.get('expected_information', ''), *queries])
    focus_terms, question_terms = terms(focus), terms(question)
    wanted = set(request.get('knowledge_task_ids', []))
    related = set(request.get('evidence_ids', []))

    canonical, aliases, hash_owner = {}, {}, {}
    for ident, doc in documents.items():
        owner = hash_owner.get(doc['sha256']) if processor.enabled else None
        if owner is None:
            owner = ident
            hash_owner[doc['sha256']] = ident
            aliases[ident] = []
        else:
            aliases[owner].append(ident)
        canonical[ident] = owner

    # Only a previously visible, exact source quote can seed an old anchor.
    anchors, opposing, anchor_quotes = {}, set(), []
    for citation in citations:
        ident, quote = citation.get('evidence_id'), citation.get('quote')
        if (ident not in documents or not citation.get('source_verified') or not citation.get('visible')
                or not isinstance(quote, str) or not quote.strip()
                or not any(quote in s['text'] for s in previous.get('visible', {}).get(ident, []))
                or citation.get('model_reported_support') == 'unsupported'):
            continue
        owner = canonical[ident]
        anchor_quotes.append((ident, quote))
        shown = next(s for s in previous['visible'][ident] if quote in s['text'])
        pos = shown['start'] + shown['text'].find(quote)
        # Old anchors reuse already delivered context. Merely growing their
        # margins must not masquerade as gap-directed new evidence.
        anchors.setdefault(owner, []).append((max(shown['start'], pos-200), min(shown['end'], pos+len(quote)+200)))
        if citation.get('model_reported_support') == 'contradicted':
            opposing.add(owner)
    for conflict in conflicts:
        if isinstance(conflict, dict):
            ids = conflict.get('evidence_ids', [])
            if isinstance(ids, list):
                opposing.update(canonical[i] for i in ids if isinstance(i, str) and i in canonical)
    related = {canonical[i] for i in related if i in canonical}

    # Reuse the ordinary extractor's boundary handling. Items are fragments, so a
    # long cited source can carry an old anchor and a new gap interval together.
    items = []
    owners = list(aliases)
    for ident in owners:
        doc = documents[ident]
        ranges = processor._ranges(doc['text'], focus, min(600, processor.span_chars), []) if processor.enabled else [(0, len(doc['text']))]
        if processor.enabled:
            seen = [(s['start'], s['end']) for i, spans in previous.get('visible', {}).items()
                    if documents[i]['sha256'] == doc['sha256'] for s in spans]
            if seen and not subtract_ranges(ranges, seen):
                alternatives = []
                for a, b in subtract_ranges([(0, len(doc['text']))], seen):
                    rr = [(a+x, a+y) for x, y in processor._ranges(doc['text'][a:b], focus, min(600, processor.span_chars), [])]
                    relevance = len(terms(' '.join(doc['text'][x:y] for x, y in rr)) & focus_terms)
                    if relevance:
                        alternatives.append((relevance, rr))
                if alternatives:
                    ranges = max(alternatives, key=lambda item: item[0])[1]
        variants = [('anchor', [r]) for r in merge_ranges(anchors.get(ident, []))]
        variants.append(('gap', ranges))
        for kind, intervals in variants:
            if not intervals:
                continue
            intervals = merge_ranges(intervals) if processor.enabled else [(0, len(doc['text']))]
            text = SEPARATOR.join(doc['text'][a:b] for a, b in intervals)
            provisional = {'views': [{'id': ident}], 'visible': {ident: [
                {'start': a, 'end': b, 'text': doc['text'][a:b]} for a, b in intervals]}}
            new = visible_delta(provisional, previous, documents)['characters']
            covered = set().union(*(set(targets.get(i, [])) for i in [ident, *aliases[ident]]))
            items.append({'id': ident, 'ranges': intervals, 'kind': kind, 'new_chars': new,
                          'opposing': ident in opposing, 'related': ident in related,
                          'focus': len(terms(text) & focus_terms),
                          'question': len(terms(text) & question_terms), 'targets': covered,
                          'text_terms': terms(text)})

    selected, selection_records = {}, []

    def cost(ranges):
        return sum(b-a for a, b in ranges) + len(SEPARATOR)*max(0, len(ranges)-1)

    def add(item):
        ident = item['id']
        updated = merge_ranges([*selected.get(ident, []), *item['ranges']])
        total = sum(cost(r) for i, r in selected.items() if i != ident) + cost(updated)
        if (ident not in selected and len(selected) >= processor.max_documents) or total > processor.max_chars:
            return False
        if updated == selected.get(ident):
            return True
        selected[ident] = updated
        selection_records.append({'source_id': ident, 'action': 'handoff_'+item['kind'],
                                  'ranges': updated, 'reported_conflict': item['opposing']})
        return True

    def rank(item):
        return (item['opposing'], item['focus'], len(item['targets'] & wanted),
                item['related'], item['question'], -cost(item['ranges']))

    old = sorted((i for i in items if i['kind'] == 'anchor'), key=rank, reverse=True)
    fresh = sorted((i for i in items if i['kind'] == 'gap' and i['new_chars']), key=rank, reverse=True)
    # Reserve actual novel text rather than a fixed fraction/document quota.
    # An unavoidable one-document/tiny budget conflict is exposed in the audit.
    if old:
        add(old[0])
    reserved = next((i for i in fresh if add(i)), None)
    if fresh and reserved is None:
        selected.clear()
        selection_records.clear()
        reserved = next((i for i in fresh if add(i)), None)
        if old:
            add(old[0])

    remaining = list(items)
    no_increment = set()
    while remaining:
        selected_terms = set().union(*(terms(documents[i]['text'][a:b]) for i, rr in selected.items() for a, b in rr)) if selected else set()
        covered = set().union(*(set(targets.get(alias, [])) for i in selected for alias in [i, *aliases[i]])) if selected else set()

        def marginal(item):
            overlap = len(item['text_terms'] & selected_terms) / max(1, len(item['text_terms']))
            return (item['opposing'], len(item['targets']-covered), item['focus'],
                    item['kind'] == 'anchor', item['related'], -overlap,
                    item['question'], -cost(item['ranges']))
        item = max(remaining, key=marginal)
        remaining.remove(item)
        adds_terms = bool((item['text_terms'] & (focus_terms | question_terms))-selected_terms)
        if (item['kind'] == 'anchor' or item['opposing'] or item['related'] or item['new_chars']
                or (item['targets']-covered) or adds_terms):
            add(item)
        else:
            no_increment.add(item['id'])

    views, visible = [], {}
    for ident, ranges in selected.items():
        doc = documents[ident]
        spans = [{'id': ident+':'+sha(doc['sha256']+':'+str(a)+':'+str(b))[:12],
                  'start': a, 'end': b, 'text': doc['text'][a:b]} for a, b in ranges]
        view = {'id': ident, 'title': doc.get('title', ident), 'text': SEPARATOR.join(s['text'] for s in spans)}
        if ranges != [(0, len(doc['text']))]:
            view['has_more'] = True
        if aliases[ident]:
            view['aliases'] = aliases[ident]
        views.append(view)
        visible[ident] = spans
        for alias in aliases[ident]:
            visible[alias] = deepcopy(spans)
            selection_records.append({'source_id': alias, 'action': 'exact_duplicate', 'canonical': ident})
    omitted = [i for i in documents if i not in visible]
    selection_records.extend({'source_id': i, 'action': 'deferred_no_marginal_signal' if canonical[i] in no_increment
                              else 'deferred_context_budget'} for i in omitted)
    similar = []
    for index, left in enumerate(views):
        a = terms(left['text'])
        for right in views[index+1:]:
            b = terms(right['text'])
            similarity = len(a & b)/math.sqrt(len(a)*len(b)) if a and b else 0
            if similarity >= .8:
                similar.append({'sources':[left['id'],right['id']], 'lexical_cosine':round(similarity,4),
                                'action':'retained_both'})
    pack = {'version': VERSION, 'views': views, 'visible': visible, 'omitted': omitted,
            'candidate_coverage': {i:sorted(set(targets.get(i, []))) for i in documents},
            'records': selection_records, 'similar_groups': similar,
            'raw_chars': sum(len(d['text']) for d in documents.values()),
            'view_chars': sum(len(v['text']) for v in views), 'cpu_seconds': time.monotonic()-started}
    validate_view(pack, documents)
    delta = visible_delta(pack, previous, documents)
    missing_anchors = []
    for ident, quote in anchor_quotes:
        if not any(quote in s['text'] for s in visible.get(ident, [])):
            missing_anchors.append(ident)
    pack['handoff'] = {'new_visible_ids': delta['evidence_ids'], 'new_visible_chars': delta['characters'],
                       'visible_ids': [v['id'] for v in views],
                       'omitted_anchor_ids': sorted(set(missing_anchors)),
                       'omitted_no_marginal_signal_ids': sorted(i for i in omitted if canonical[i] in no_increment),
                       'conflict_ids': sorted(opposing), 'budget_unit': 'characters',
                       'omitted_conflict_ids': sorted(i for i in opposing if i not in visible),
                       'selection_note': '已有引用/冲突标签与词面排序是候选信号，不代表医学支持或完整覆盖。'}
    return pack
