"""Attribute saved Agent outputs without changing whitespace or invoking models."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from audit_medqa_payloads import attribute


def category(role, path):
    if not path:return 'json_layout'
    top,last=path[0],path[-1]
    if role=='Planner':
        if top=='memory_assessments':return 'memory_reasons' if last=='reason' else 'memory_choices_and_ids'
        if top=='key_conditions':return 'key_conditions'
        if top=='knowledge_tasks':return 'knowledge_tasks'
        if top=='criteria':return 'criteria'
    if role=='Investigator':
        if top in {'summary','findings','memory_summary','evidence_summary','analysis','reasoning'}:
            return 'extra_summaries_or_analysis'
        if last=='query':return 'queries'
        if last=='expected_information':return 'expected_information'
        if last=='gap' or top=='gaps':return 'gaps'
        if top=='search':return 'tool_plan_fields'
    if role=='VerifierReasoner':
        if top=='final_answer':return 'answer_summary' if last=='summary' else 'answer_fields'
        if top=='citations':
            return 'citation_quotes' if last=='quote' else 'citation_reasons' if last=='reason' else 'citation_refs_and_status'
        if top=='critical_gaps':return 'critical_gaps'
        if top in {'claims','checks','condition_checks','option_comparison','analysis','reasoning'}:
            return 'extra_audit_or_analysis'
    return 'other_fields'


class OutputSpans:
    """A read-only JSON scanner retaining every original character, including junk."""
    def __init__(self,text,role):
        self.text,self.role,self.pos,self.spans=text,role,0,[]
        self.decoder=json.JSONDecoder()

    def take(self,end,path):
        if end>self.pos:self.spans.append((self.pos,end,category(self.role,path)))
        self.pos=end

    def space(self,path):
        end=self.pos
        while end<len(self.text) and self.text[end].isspace():end+=1
        self.take(end,path)

    def value(self,path=()):
        self.space(path)
        token=self.text[self.pos]
        if token=='{':
            self.take(self.pos+1,path);self.space(path)
            if self.text[self.pos]=='}':self.take(self.pos+1,path);return
            while True:
                self.space(path)
                key,end=self.decoder.raw_decode(self.text,self.pos)
                if not isinstance(key,str):raise ValueError('invalid key')
                child=(*path,key);self.take(end,child);self.space(child)
                if self.text[self.pos]!=':':raise ValueError('missing colon')
                self.take(self.pos+1,child);self.value(child);self.space(path)
                if self.text[self.pos]=='}':self.take(self.pos+1,path);return
                if self.text[self.pos]!=',':raise ValueError('missing comma')
                self.take(self.pos+1,path)
        elif token=='[':
            self.take(self.pos+1,path);self.space(path)
            if self.text[self.pos]==']':self.take(self.pos+1,path);return
            index=0
            while True:
                self.value((*path,index));self.space(path);index+=1
                if self.text[self.pos]==']':self.take(self.pos+1,path);return
                if self.text[self.pos]!=',':raise ValueError('missing comma')
                self.take(self.pos+1,path)
        else:
            _,end=self.decoder.raw_decode(self.text,self.pos);self.take(end,path)

    def scan(self):
        malformed=False
        try:self.value();self.space(())
        except (ValueError,IndexError):malformed=True
        if self.pos<len(self.text):
            malformed=True
            self.spans.append((self.pos,len(self.text),'unparsed_tail'));self.pos=len(self.text)
        assert sum(end-start for start,end,_ in self.spans)==len(self.text)
        return self.spans,malformed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    report=json.loads((args.run/'report.json').read_text())
    result={'model_calls_during_analysis':0,'run':str(args.run.resolve()),
        'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        'method':'Original response bytes retained; JSON path spans; local token offset majority attribution; boundary ownership is a convention.',
        'conditions':{}}
    for condition in report:
        totals=defaultdict(Counter);counts=Counter();details=[];keys=defaultdict(Counter)
        for row in map(json.loads,(args.run/(condition+'.jsonl')).read_text().splitlines()):
            for event in row['events']:
                if 'agent' not in event or 'response' not in event:continue
                role,text=event['agent'],event['response']
                spans,malformed=OutputSpans(text,role).scan()
                tokens,boundaries=attribute(tokenizer,text,spans)
                assert sum(tokens.values())==event['usage']['completion_tokens'],(condition,row['question_id'],role)
                totals[role].update(tokens);counts[role]+=1
                obj=event.get('output',{})
                keys[role].update(obj.keys())
                details.append({'question_id':row['question_id'],'role':role,'tokens':dict(tokens),
                    'malformed':malformed,'cross_category_boundary_tokens':boundaries})
        for role,items in totals.items():
            assert sum(items.values())==report[condition]['role_usage'][role]['completion_tokens']
        result['conditions'][condition]={'calls_by_role':dict(counts),'output_tokens_by_role':dict(totals),
            'observed_output_keys':dict(keys),'calls':details}
        print(condition,json.dumps(dict(totals),ensure_ascii=False))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
