# 当前纯文本与结构化通信的设计与结果解释

分析日期：2026-10-20；只读取现有71题四条件结果，无API调用、无实现变更。源码hash与原manifest一致。

|比较|纯文本总token|结构化总token|结构化相对变化|通信字符相对变化|
|---|---:|---:|---:|---:|
|无共享记忆|372,471|358,295|少3.81%|少2.85%|
|有共享记忆|457,940|451,138|少1.49%|多1.16%|

四条件各213次模型调用、426条请求/响应计数，均使用deepseek-flash。结构化无记忆有1次JSON输出解析失败，正式比较标志因而关闭；表中为完整usage和字符的观察值。

## 实际实现

LangChain/medqa/runtime.py:50的compact用json.dumps(ensure_ascii=False,separators=(",",":"))；:54的render_text将同一个dict/list树递归渲染为字段名、冒号、缩进、换行和列表。:92仅根据mode选择这两个函数，然后送入相同角色SystemMessage和HumanMessage。输出两种模式都要求JSON，:945同用json.loads解析。

MedQA路径通过g2cp_profile.message调用公共g2cp构造器，分Control（调度身份、动作、能力与步数）、Semantic（该profile当前只有memory_ids等references）和Precise（角色所需输入）。只有Precise进入模型。两种模式共用这个运行时和三层承载，也共用按角色选字段、Evidence Processor、预算、校验与记忆策略。模型输出先被Runtime解析并更新状态，下一角色的输入由程序重新组装；没有把整份上一个响应原样传给下一个角色。

所以当前text是“同字段的可读文本输入”，structured是“同字段的紧凑JSON输入”。两者的输出均为JSON。现实验无法单独证明整个G2CP承载、ID引用、按需取状态比自由自然语言协作节省多少，因为这些机制被两组共同使用。

## 固定内容的格式重放

对四条件全部852条已记录模型请求，以原输入对象分别渲染text和JSON。纯格式输入字符减少3.68%、3.68%、4.40%、4.41%（各自轨迹）；详细数值在format-replay.json。按实际模式重算的输入字符与report.request_chars四组均完全一致。

简单Planner输入的JSON甚至每条多3字符，不能声称JSON天然总是更省。没有使用DeepSeek实际tokenizer测这些反事实输入，因此字符减少不是token减少。重放没有新模型输出，不能测试质量差异。

原始在线结果还包含轨迹变化：无记忆text/structured搜索90/94次，有记忆83/94次；选择的文献、输出长度和在线形成的记忆库随之变化。相同Planner输入在部分调用中也有不同输出。因此总token差异既有表示格式因素，也有模型/检索/记忆轨迹波动。

## 统计口径

communication_chars是角色HumanMessage正文字符加原始模型响应字符，不含SystemMessage、Control或Semantic元数据、API网络封装。system_chars、protocol_control_bytes另列。总token是API返回的prompt_tokens+completion_tokens，包含模型请求中系统提示等费用口径；缓存token已经包含在prompt中。messages每请求/响应各加1，是Runtime调用计数，不是操作系统socket包数。非文本向量负载另计state_vector_transfers/state_bytes；本实验的输入格式对比不等于隐藏状态传入聊天模型。

这批全量数据能报告固定配置下的端到端差异，以及离线同内容字符压缩；不能把全部差异归因于格式或推广成完整协议的因果收益。后续若扩展实验，应分别控制内容选择、表示格式和模型执行轨迹，并保留原始结果。
