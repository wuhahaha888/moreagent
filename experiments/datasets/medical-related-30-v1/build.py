"""Extract the user's exact three MedQA sequences; no model calls or runtime edits."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
GROUPS = [
    ("medqa-anemia", "贫血分类、检验解释与条件变化",
     ["0900", "0418", "1225", "0394", "1009", "0409", "0476", "0810", "0673", "1205"]),
    ("medqa-renal", "蛋白尿、血尿来源与肾脏机制鉴别",
     ["0574", "0145", "0169", "0533", "0697", "0162", "0465", "0216", "1153", "1094"]),
    ("medqa-thyroid", "甲状腺功能、母婴条件与用药边界",
     ["0067", "0361", "0089", "0512", "0137", "1061", "0103", "0124", "0229", "0396"]),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def build(output):
    source_path = PROJECT.parent / "MedRGAG/benchmark.json"
    selection_path = HERE / "selection.json"
    selection = json.loads(selection_path.read_text())
    source_raw = source_path.read_bytes()
    source_sha = sha(source_raw)
    source = json.loads(source_raw)["medqa"]
    assert source_sha == selection["source_sha256"], "Original source fingerprint changed"
    parent_groups = {g["group_id"]: g for g in selection["groups"]}
    reviews = {r["task_id"]: r for r in selection["candidate_reuse_annotations"]}
    all_ids = [i for _, _, ids in GROUPS for i in ids]
    assert len(all_ids) == len(set(all_ids)) == 30
    public, tasks, answers, groups, annotations, runner = [], [], {}, [], [], {}
    prior_ids = set()
    for group_position, (gid, name, ids) in enumerate(GROUPS, 1):
        assert ids == parent_groups[gid]["original_question_ids"], (gid, "Sequence mismatch")
        group_task_ids = []
        prior_in_group = set()
        for round_in_group, original_id in enumerate(ids, 1):
            original = source[original_id]
            task_id = "medqa::" + original_id
            question, options, answer = original["question"], original["options"], original["answer"]
            assert len(options) == 4 and answer in options
            payload_sha = sha(json.dumps({"question": question, "options": options},
                                        sort_keys=True, ensure_ascii=False).encode())
            row = {"question_id": task_id, "question": question, "options": options}
            public.append(row)
            tasks.append({**row, "answer": answer, "source_dataset": "medqa",
                          "source_question_id": original_id, "group_id": gid,
                          "group_position": group_position, "round_in_group": round_in_group,
                          "global_position": len(tasks) + 1, "question_options_sha256": payload_sha})
            answers[task_id] = {"answer": answer, "answer_text": options[answer],
                                "source_question_id": original_id,
                                "question_options_sha256": payload_sha}
            runner[task_id] = {"question": question, "options": options, "answer": answer,
                               "source_dataset": "medqa", "source_question_id": original_id}
            annotation = dict(reviews[task_id])
            assert annotation["group_id"] == gid
            assert set(annotation["reuse_from_task_ids"]) <= prior_in_group
            annotation["relation_status"] = "candidate_hypothesis_not_observed_reuse"
            annotations.append(annotation)
            prior_in_group.add(task_id)
            prior_ids.add(task_id)
            group_task_ids.append(task_id)
        groups.append({"group_id": gid, "name": name, "group_position": group_position,
                       "question_count": 10, "original_question_ids": ids,
                       "task_ids": group_task_ids})

    assert all(set(r) == {"question_id", "question", "options"} for r in public)
    output.mkdir(parents=True, exist_ok=True)
    generated = []

    def write(name, raw):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        generated.append(name)

    def save(name, value):
        raw = encoded(value)
        write(name, raw)
        return sha(raw)

    def jsonl(name, rows):
        write(name, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())

    jsonl("questions.jsonl", public)
    jsonl("evaluation/tasks.jsonl", tasks)
    save("evaluation/answers.json", answers)
    save("evaluation/reuse-review.json", {
        "annotation_role": "evaluator_only_do_not_preload_into_agents_or_memory",
        "relationship_type": "candidate_knowledge_reuse_not_required_task_dependency",
        "tasks": annotations})
    runner_sha = save("runner-benchmark.json", {
        "_metadata": {"dataset_name": "medical-related-30-v1", "source_dataset": "medqa",
                      "answer_policy": "loader_and_grader_only; Question.public excludes gold"},
        "medqa": runner})
    order = [r["question_id"] for r in public]
    save("order.json", {"dataset_sha256": runner_sha, "ids": order,
                        "ordering": "user_fixed_anemia_renal_thyroid_preserve_within_group"})
    save("groups.json", {"annotation_role": "evaluator_only", "groups": groups})
    text = ["# 三组医学关联任务：30 道原题（不含答案）", "",
            "顺序：贫血 → 肾脏 → 甲状腺，每组 10 题。原题英文及选项保持原样。", ""]
    for group in groups:
        text.extend(["## " + group["group_id"] + "：" + group["name"], ""])
        group_rows = [r for r in public if r["question_id"] in group["task_ids"]]
        jsonl("groups/" + group["group_id"] + ".questions.jsonl", group_rows)
        save("groups/" + group["group_id"] + ".order.json", {
            "dataset_sha256": runner_sha, "ids": group["task_ids"]})
        for n, row in enumerate(group_rows, 1):
            text.extend([f"### 第 {n} 轮 · {row['question_id']}", "", row["question"], ""])
            text.extend(f"- **{letter}.** {content}" for letter, content in row["options"].items())
            text.append("")
    write("questions.md", ("\n".join(text) + "\n").encode())
    table = io.StringIO(newline="")
    fields = ["group_id", "round_in_group", "question_id", "source_question_id", "question", "A", "B", "C", "D", "question_options_sha256"]
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    for row in tasks:
        writer.writerow({**{k: row[k] for k in fields if k not in "ABCD"}, **row["options"]})
    write("questions.csv", table.getvalue().encode("utf-8-sig"))
    save("metadata.json", {
        "dataset_name": "medical-related-30-v1", "construction_date": "2026-10-20",
        "source_path": str(source_path), "source_sha256": source_sha,
        "parent_subset": selection["parent_subset"], "parent_subset_question_count": 71,
        "selection_manifest_sha256": sha(selection_path.read_bytes()),
        "question_count": 30, "group_count": 3, "source_dataset": "medqa",
        "group_order": [g["group_id"] for g in groups], "ordering": "user_fixed_no_shuffle",
        "original_question_options_and_gold_preserved": True,
        "selection_basis": "exact user supplied IDs and order; no new outcome-based selection",
        "prior_exposure": "all 30 appeared in the existing 71-question experiments",
        "evaluation_status": "new curated evaluation subset, not an unseen independent test set",
        "continuity_type": "knowledge-reuse task streams, not required predecessor dependencies",
        "runtime_changed": False, "existing_configs_changed": False, "model_calls": 0})
    assert sha(source_path.read_bytes()) == source_sha
    save("validation.json", {
        "question_count": len(public), "unique_ids": len(prior_ids), "group_count": len(groups),
        "questions_per_group": {g["group_id"]: g["question_count"] for g in groups},
        "exact_user_ids_and_order_verified": True,
        "exact_original_question_options_and_answer_verified": True,
        "public_questions_contain_only_id_question_options": True,
        "candidate_reuse_edges_point_only_to_earlier_same_group_tasks": True,
        "source_benchmark_unchanged": True, "runner_sha256": runner_sha, "model_calls": 0})
    readme = """# 医学关联任务子集：三组各十题

按用户确认的 ID 提取原始 MedQA：贫血 → 肾脏 → 甲状腺，组内顺序固定。全局 30 轮，不打乱题干、选项或答案。

- `questions.md`：供人阅读的完整 30 道英文原题与选项，不含答案。
- `questions.jsonl`、`questions.csv`：不含答案的题目导出。
- `runner-benchmark.json`：现有加载器兼容文件，包含供评分器使用的标准答案；运行时 `Question.public()` 只暴露 ID、题干和选项。
- `order.json`：30 题固定顺序，指纹绑定新的 runner 文件。
- `groups/`：各组十题的无答案文件及顺序文件；每份顺序文件仍绑定根目录同一个 runner。
- `evaluation/answers.json`、`evaluation/tasks.jsonl`：仅供评分与数据管理。
- `evaluation/reuse-review.json`：继承的候选复用关系与待核对条件，仅供评估者使用，不得预灌到 Agent 或共享记忆；其中的候选关系不是实测复用记录。
- `groups.json`、`metadata.json`、`validation.json`、`checksums.json`：分组、来源、原文指纹与校验记录。
- `selection.json`：用户固定的题号、组内顺序及候选关联注释；重建无需旧 71 题目录。

这 30 题均已出现在此前 71 题实验中，是新整理的评测子集，不是全新未见的独立测试集。仅有主题和顺序不能证明有效复用，后续实验仍须记录历史来源、回读、实际使用、正确性及成本。

使用现有 moreagent 加载器时，将新实验配置的 `benchmark` 指向本目录 `runner-benchmark.json`，`ids` 指向本目录 `order.json`，并设 `split=all`。三组独立实验分别采用 `groups/<group_id>.order.json`，使用新的结果目录及记忆命名空间；需在方案中声明组间是否清空记忆。

数据包构建不调用模型，不修改运行时、实验结果或记忆库。当前四条件 API 配置见 `configs/medical-related-30-v1`。

复现：在 moreagent 根目录运行 `python -B experiments/datasets/medical-related-30-v1/build.py`。
"""
    write("README.md", readme.encode())
    checksums = {name: sha((output / name).read_bytes()) for name in generated}
    checksums["selection.json"] = sha(selection_path.read_bytes())
    checksums["build.py"] = sha(Path(__file__).read_bytes())
    save("checksums.json", checksums)
    print(json.dumps({"output": str(output), "questions": 30, "groups": [g["group_id"] for g in groups],
                      "runner_sha256": runner_sha, "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output.resolve())
