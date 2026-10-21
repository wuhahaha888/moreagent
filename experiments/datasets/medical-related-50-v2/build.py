"""Build a reproducible 30-original + 20-originally-authored challenge stream.

No API calls. Answers, provenance and reuse hypotheses stay evaluator-side.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import runpy

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'medical-related-30-v1'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()


def build(output):
    source_raw=(BASE/'runner-benchmark.json').read_bytes()
    original=json.loads(source_raw)['medqa']
    original_order=json.loads((BASE/'order.json').read_text())['ids']
    assert len(original_order)==len(original)==30
    sources=json.loads((HERE/'evaluation/sources.json').read_text())
    items=runpy.run_path(str(HERE/'evaluation/items.py'))['ITEMS']
    assert len(items)==20 and len({i['question_id'] for i in items})==20
    assert Counter(i['group_id'] for i in items)=={'anemia':7,'renal':7,'thyroid':6}
    target_letters=list('A'*5+'B'*5+'C'*5+'D'*5)
    random.Random(20261021).shuffle(target_letters)
    runner={k:dict(original[k]) for k in original_order}
    answers={k:{'answer':r['answer'],'answer_text':r['options'][r['answer']],
                'provenance':'original_medqa_unchanged'} for k,r in runner.items()}
    seen=set(original_order);reviews=[];lengths=[]
    for index,(row,answer) in enumerate(zip(items,target_letters)):
        ident=row['question_id'];assert ident not in seen
        assert set(row['parents'])<=seen and len(row['parents'])==len(set(row['parents']))
        assert len(set(row['parents']) & set(original_order))>=2
        assert len(row['question'].split())>=70
        assert all(k in sources['sources'] for k in row['source_keys'])
        assert len(row['distractors'])==3 and all(d['why_wrong_zh'] for d in row['distractors'])
        choices=list(row['distractors'])
        random.Random(20261021+index).shuffle(choices)
        choices.insert('ABCD'.index(answer),{'text':row['correct'],'why_wrong_zh':None})
        options={letter:c['text'] for letter,c in zip('ABCD',choices)}
        assert len(set(options.values()))==4
        runner[ident]={'question':row['question'],'options':options,'answer':answer,
                      'source_dataset':'moreagent_medqa_style_authored_extension','source_question_id':ident}
        answers[ident]={'answer':answer,'answer_text':row['correct'],'rationale_zh':row['rationale_zh'],
            'distractor_rationales_zh':{k:c['why_wrong_zh'] for k,c in zip('ABCD',choices) if k!=answer},
            'source_keys':row['source_keys'],'independent_clinical_review':'pending',
            'difficulty':'designed_multi_step; not_empirically_calibrated'}
        reviews.append({'question_id':ident,'group_id':row['group_id'],'global_position':31+index,
            'reuse_from_task_ids':row['parents'],'continuity_types':row['continuity_types'],
            'knowledge_needs':[{'id':'N1','goal_zh':row['reusable_knowledge_zh'],'candidate_sources':row['parents'],
                                'status':'reuse_only_if_actual_stored_evidence_covers_current_scope'},
                               {'id':'N2','goal_zh':row['remaining_gap_zh'],'status':'look_up_if_not_already_covered'}],
            'conditions_to_recheck':row['negative_transfer_zh'],
            'relationship_type':'candidate_knowledge_reuse; not_a_required_patient_followup',
            'annotation_role':'evaluator_only; never_preload_into_agents_or_memory'})
        seen.add(ident);lengths.append(len(row['question'].split()))
    output.mkdir(parents=True,exist_ok=True)
    files=[]
    def write(name,raw):
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);files.append(name)
    def save(name,value):
        raw=encoded(value);write(name,raw);return sha(raw)
    runner_sha=save('runner-benchmark.json',{'_metadata':{
        'dataset_name':'medical-related-50-v2','original_count':30,'authored_count':20,
        'answer_policy':'grader_only; Question.public exposes id/question/options only'},'medqa':runner})
    order=list(runner);extension=[i['question_id'] for i in items]
    save('order.json',{'dataset_sha256':runner_sha,'ids':order,'ordering':'unchanged_original_30_then_authored_20'})
    save('extension.order.json',{'dataset_sha256':runner_sha,'ids':extension,
        'warning':'Not a warm-memory experiment by itself; use full 50 or an explicitly frozen matched warmup database.'})
    public=[{'question_id':k,'question':r['question'],'options':r['options']} for k,r in runner.items()]
    write('questions.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in public).encode())
    def questions_md(rows,title):
        lines=[title,'','仅题目与选项；新增题为原创衍生题，非MedQA官方新增试题。','']
        for position,r in rows:
            lines.extend([f"## {position} · {r['question_id']}",'',r['question'],''])
            lines.extend(f"- **{k}.** {v}" for k,v in r['options'].items());lines.append('')
        return ('\n'.join(lines)+'\n').encode()
    write('questions.md',questions_md(enumerate(public,1),'# 50题连续流：原始30题＋原创扩展20题'))
    write('extensions.md',questions_md(enumerate(public[30:],31),'# 新增20道MedQA风格衍生题'))
    save('evaluation/answers.json',answers)
    save('evaluation/reuse-review.json',{'annotation_role':'evaluator_only','tasks':reviews})
    guide=['# 扩展题答案与连续性审核（仅评估侧）','',
           '全部为可复用知识关系假设；不保证原任务实际保存了所需引句。不得预加载答案或此表到运行时。','',
           '| 新题 | 组 | 正确选项 | 前序知识来源 | 需重新核对的条件 |','|---|---|---|---|---|']
    for row,review in zip(items,reviews):
        ident=row['question_id']
        guide.append('| '+ ' | '.join([ident,row['group_id'],answers[ident]['answer'],
            ', '.join(row['parents']),row['negative_transfer_zh']])+' |')
    for row in items:
        ident=row['question_id'];record=answers[ident]
        guide.extend(['',f'## {ident}','',f"答案：{record['answer']} — {record['answer_text']}",'',
            '可复用：'+row['reusable_knowledge_zh'],'','仍需核实：'+row['remaining_gap_zh'],'',
            '解释：'+row['rationale_zh'],''])
        for letter,why in record['distractor_rationales_zh'].items():guide.append(f'- {letter}：{why}')
        guide.append('')
        for key in row['source_keys']:
            source=sources['sources'][key];guide.append(f"- [{source['title']}]({source['url']})")
    write('evaluation/continuity.md',('\n'.join(guide)+'\n').encode())
    save('slices.json',{'original30':original_order,'extension20':extension,
        **{f'extension_{g}':[i['question_id'] for i in items if i['group_id']==g] for g in ('anemia','renal','thyroid')}})
    metadata={'dataset_name':'medical-related-50-v2','created':'2026-10-21','question_count':50,
        'original_count':30,'authored_count':20,'original_runner_sha256':sha(source_raw),
        'runner_sha256':runner_sha,'authored_group_counts':dict(Counter(i['group_id'] for i in items)),
        'order_policy':'original 30 unchanged first; authored A01-A07,R01-R07,T01-T06 appended',
        'continuity':'cross_task_knowledge_reuse_and_condition_changes; all cases self_contained',
        'patient_continuity':'different synthetic patients unless explicitly stated; no fabricated shared identity',
        'evaluation_status':'development challenge set; no empirical difficulty calibration or independent clinical review',
        'model_calls_during_construction':0,'outcome_based_selection':False,
        'web_sources':'supporting factual concepts; not injected into retrieval corpus or memories',
        'memory_experiment':'same corpus/questions/budgets in all four conditions; independent cold databases; online accumulation',
        'answer_letter_counts_extension':dict(Counter(answers[k]['answer'] for k in extension)),
        'new_question_word_range':[min(lengths),max(lengths)]}
    save('metadata.json',metadata)
    save('validation.json',{'original_question_options_gold_preserved':all(runner[k]==original[k] for k in original_order),
        'original_order_preserved_as_prefix':order[:30]==original_order,'count':len(order),'unique_ids':len(set(order)),
        'new_reuse_edges_only_to_earlier_tasks':all(set(r['reuse_from_task_ids'])<=set(order[:r['global_position']-1]) for r in reviews),
        'new_items_have_at_least_two_original_anchors':True,'new_items_have_three_explained_distractors':True,
        'public_fields':['question_id','question','options'],'new_answer_balance':dict(Counter(answers[k]['answer'] for k in extension)),
        'difficulty_measured':False,'independent_clinical_review':'pending','paid_model_calls':0})
    readme='''# 原始MedQA 30题＋原创衍生20题

原始30题的题干、选项、答案和前30轮顺序均保持不变，之后追加贫血7题、肾脏7题、甲状腺6题。新增为本项目原创的MedQA风格合成病例，不是官方MedQA题；难度仅为设计目标，尚未经独立医学审题和盲测，不能保证准确率低于100%。不按实验成败删选题目。

- `extensions.md`：可先查看的新增20题，无答案。
- `questions.md` / `questions.jsonl`：完整50题，无答案和复用提示。
- `runner-benchmark.json` / `order.json`：兼容现有加载器；答案仅供评分器，Agent通过Question.public只收到题目、选项和ID。
- `evaluation/answers.json` / `evaluation/continuity.md`：答案、干扰项解释及医学参考来源，评估侧专用。
- `evaluation/reuse-review.json`：前序知识、适用性复核与新增缺口，都是待验证的复用假设，不得预灌Agent或记忆。
- `evaluation/items.py` / `evaluation/sources.json`：原创编制源文件及来源目录；build.py离线重建。
- `slices.json`：原30题、新20题及新增各主题的独立统计切片；应分别报告，避免前30题高分稀释挑战题表现。

连续性通过知识证据的复用、组合、条件反转及延迟回读体现；病例本身保持自足，无记忆条件仍可从相同教材重新检索。前序任务若没有实际保存相应证据，不能仅按题号关系宣称有效命中。没有通过改写病人身份或隐藏关键事实制造前置依赖。

运行全50题才会自然完成原30题的在线记忆预热。extension.order.json单独执行会冷启动，除非另有严格匹配的冻结预热库，不得称为同等连续性实验。组间不清空记忆；四条件数据库独立。

新增题包含混合机制、检验动态、病理与暴露交叉、错误迁移负例。高难题不等于强制检索更多；记忆需要先通过条件复核，再决定哪些关系已覆盖、哪些关系需补查。选项正确率、证据质量、错误复用、检索/调用次数和总Token分别报告。

新配置位于configs/medical-related-50-v2。重建：`python -B experiments/datasets/medical-related-50-v2/build.py`。旧数据、旧配置和旧结果不覆盖。
'''
    write('README.md',readme.encode())
    manifest={name:sha((output/name).read_bytes()) for name in files}
    for name in ('build.py','evaluation/items.py','evaluation/sources.json'):
        manifest[name]=sha((HERE/name).read_bytes())
    save('checksums.json',manifest)
    assert (BASE/'runner-benchmark.json').read_bytes()==source_raw
    print(json.dumps({'output':str(output),'questions':50,'authored':20,'runner_sha256':runner_sha,'paid_api_calls':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE)
    build(parser.parse_args().output.resolve())
