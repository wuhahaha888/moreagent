# 原始MedQA 30题＋原创衍生20题

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
