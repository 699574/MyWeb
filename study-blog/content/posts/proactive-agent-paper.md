---
title: 'Proactive Agents'
date: '2025-12-17'
tags: ['Agent']
---

最近在做一些有关对话式搜索的工作，里面有挺多和Proactive Agent相关的部分，花了一天整体调研了一下这方面的文章，大概总结一下：

### Module and Mechanism

Ask-before-Plan: Proactive Language Agents for Real-World Planning(EMNLP 2024)
 - 用户指令模糊或不可行，现有研究主要集中于澄清或规划某一特定方面，且澄清机制缺乏环境反馈，依赖模型内部推理
 - 提出CEP框架：澄清，执行，规划三个Agent之间进行协作
 - 澄清Agent接收用户对话和执行Agent结果后决定是否进行澄清询问
 - 执行Agent收集无效操作的反思并存储，用于记忆回溯。在当前轮次生成动作时复用反思
 - 数据集：TravelPlanner，构造缺失与不可行信息(拓扑排序)，构建模糊指令
 - 评估指标：
	 - 澄清需求预测：Clarification Accuracy(预测是否需要澄清的准确率)，Clearness Recall(能够正确识别“清晰指令”不需提问的召回率)，Vagueness Recall(能够正确识别“模糊指令”需要提问的召回率)
	 - 澄清问题生成：Rule-based Score（生成的问题是否包含针对缺失细节的预定义关键词），BLEU(生成问题与Ground Truth的 N-gram 重合度（最大 4-gram）), GPT Score(使用 GPT-4 (Zero-shot) 对生成问题的正确性进行评分)
	 - 执行部分: Well-formed(生成的 API 调用没有语法错误的比例),API Match(API 名称预测正确的比例),Correctness(参数预测的 Precision/Recall/F1 分数),Repeat Rate(工具调用重复率（主要针对动态执行中的死循环问题）),Pass Rate(工具调用链与 Ground Truth 完全一致的比例)
	 - 规划部分: Delivery Rate(能在有限步数内生成有效JSON计划的比例),Commonsense Constraint Pass Rate(常识约束通过率),Hard Constraint Pass Rate(硬性约束通过率),Final Pass Rate

From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?(ICML 2025)
 - 现实Agent场景通常面对信息不完全的问题
 - 提出主动推理(AR)范式，要求模型通过与外部环境进行多轮交互，消除不确定性获得解决方案。
 - 数据集：提出AR-Bench，提供三大任务：侦探破案(DC),海龟汤(SP)，猜数字(GN)
 - 评估指标：
	 - 采用动态交互式评估，被测模型扮演玩家
	 - 对于 GN 任务，使用规则脚本提供反馈，对于DC和SP任务，使用高智力模型回答。
	 - 结果指标：Accuracy(DC),Character-level F1 Score(SP),Exact Match(GN)
	 - 过程指标:采用信息覆盖率量化提问策略质量

Proactive Conversational Agents with Inner Thoughts(CHI 2025)
 - 现有对话AI在多人对话中很被动，局限于被动响应和下一发言人预测机制(NSP)
 - AI应基于内在动机发言
 - 提出基于认知科学的框架：
	- Trigger：响应新消息以及沉默/冷场
	- Retrieval：基于上下文检索记忆，引入Saliency Score(包含相似度和时间衰减)，模仿人类记忆唤醒机制
	- Thought Formation：快市场简单回应，基于检索记忆慢生成深度想法
	- Evaluation：对自己生成的想法打分
	- Participation：分数超过阈值就发言，分数低就沉默
 - 数据集：
	- Multiparty Chat Corpus：证明现有模型无法准确预测下位发言人
	- PersonaChat：构建AI人设
	- Think-aloud Study Data：构建Inner Thoughts框架的底层逻辑
 - 评估指标：
	- Prediction Accuracy：用于Multiparty Chat Corpus
	- Likert Scale：人类打分评估
	- Behavioral Identification Accuracy：用户研究部分

Reward-Driven Interaction: Enhancing Proactive Dialogue Agents through User Satisfaction Prediction
 - 目前语音助手存在两个问题：用户隐式反馈存在噪声，ASR错误导致错误语句时模型难以识别用户不满；长尾领域数据稀疏，难以学习到通用模式。
 - 引入多任务学习，增加两个辅助任务：
	- 对比自监督学习：用于解决ASR错误和稀疏语句表征问题，采用SimCSE
	- 领域-意图分类：用于解决长尾领域数据不足，使用迁移学习思路，在session编码层后增加分类头，预测对话所属的domain和intent。
 - 数据集：百度语音交互系统日志
 - 评价指标：
	- Offline：AUC，CLA(在Precision大于阈值时模型的最大Recall) 
	- Online：CUS(人工标注用户满意比例),ASR错误召回数，长尾领域召回数

FlowDelta: Modeling Flow Information Gain in Reasoning for Conversational Machine Comprehension(EMNLP 2019)
 - 模型无法区分上轮对话的旧信息和本轮对话引入的新信息
 - 在输入显式加入上轮状态和之前状态的差值
 - 可以认为是一种信息增益

### Framework

CollabLLM: From Passive Responders to Active Collaborators(ICML 2025)
 - 现有RLHF机制集中于token级别的优化，模型倾向于被动响应
 - 提出了 COLLABLLM 框架，将对话视为一个序列决策过程，类似RL的MC方法
 - 数据集：基于Medium文章数据集，BigCodeBench，MATH数据集采样构建交互式环境，在Abg-CoQA数据集上做Zero-shot测试
 - 评估指标：
	 - 任务完成度指标：BLEU(MediumDocEdit-Chat),Pass Rate(BigCodeBench-Chat),Accuracy(MATH-Chat) 
	 - 交互质量指标: Interactivity (ITR, 交互性评分),Average Token Count (Tokens, 效率)
	 - 真人评估：用户满意度，耗时，多轮评分

Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction(ACL 2025)
 - 现有研究忽略了生成回复的一致性，常常出现幻觉与不一致。
 - 提出CRC框架，首先进行一致性反思，返回不一致类型和建议后进行一致性修正
 - 数据集：DuRecDial, DuRecDial 2.0, TopDial
 - 评估指标：
	 - 文本生成质量：Word-level F1,BLEU-2,Dist-2 
	 - 任务核心指标：Knowledge F1(生成回复包含领域知识实体的正确率),Goal Success Rate(对话是否达成了预定目标动作和话题)
	 - 人工评价：User Profile Consistency,Dialogue History Consistency,Domain Knowledge Consistency,Subgoal Consistency

Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance(ICLR 2025)
 - 现有LLM Agent依赖被动响应，缺乏主动性
 - 设计一个闭环的数据生成流水线
 - 环境生成事件流和维护状态，User Agent模拟用户行为，并对Agent建议提出接受拒绝反馈，Proactive Agent实时观察事件流，预测意图决定是否发起协助
 - 训练reward model替代人类判断
 - 数据集：构建ProactiveBench
 - 评估指标：F1-score，False-Alarm，Precision，Accuracy，Recall

ConsistentChat: Building Skeleton-Guided Consistent Multi-Turn Dialogues for Large Language Models from Scratch(EMNLP 2025)
 - LLM在长对话中容易出现上下文漂移和逻辑不一致
 - 现有方法存在局限性：逐轮生成时难以兼顾全局目标，缺乏人类意图建模，长对话数据连贯性参差不齐
 - 提出一种数据生成范式：
	- 对意图进行建模，为每种意图设计特定信息流
	- 全局骨架生成：一次性生成整个对话中用户所有提问序列
	- 填充回答：在已知提问序列的基础上通过CoT生成Agent回复
 - 数据集构建：ConsistentChat
 - baseline数据集：ShareGPT，ChatAlpaca，UltraChat，Lmsys-Chat-1M
 - Benchmark：LIGHT,TOPDIAL,MT-EVAL
 - 评估方法：LLM-as-a-Judge
	 - 自动评估：Chat  Consistency Score
	 - 人工评估
	 - 通用能力评估：MMLU,MATH,HumanEval,GSM8K

ReviewInstruct: A Review-Driven Multi-Turn Conversations Generation Method for Large Language Models(ACL 2025)
 - LLM多轮对话中连贯性不足，难以持续澄清纠错
 - SFT数据缺陷：以单轮数据为主，简单拼接伪多轮对话不可行，ask-respond范式存在结构性问题
 - 提出Review-Instruct框架：
	- 在respond和下轮ask间加入review阶段
	- candidate负责回答，reviewers独立评估回答，chairman汇总反馈决策指令演化方向
	- 反馈正向时进行广度扩展，指出不足时进行深度扩展
 - 数据集：
	- 原始指令数据集：Alpaca Dataset
	- 合成多轮对话数据：Review-Instruct Multi-turn Dataset 
 - 评估指标：
	- Benchmark: MT-Bench, MMLU-Pro, Auto-Arena, 人工评测
	- 辅助指标：指令难度，多样性

STaR-GATE: Teaching Language Models to Ask Clarifying Questions(COLM 2024)
 - 人类提问常常具有模糊性
 - 提出STaR-GATE框架：
	- Q向R提问，R基于隐性persona回答，计算全知模型生成gold response的对数概率
	- 使用Q模型自己生成答案，防止出现幻觉
 - 数据集：自构建的{task，persona}对

Clarify When Necessary: Resolving Ambiguity Through Interaction with LMs(NAACL 2025)
 - 框架：标准的决策-澄清-回答
 - 创新点：INTENT-SIM方法：
	- 生成澄清问题
	- 模拟用户回答，采样生成S个
	- 针对每个模拟回答，让模型生成最终预测
	- 使用NLI模型判断不同回答是否语义等价，归类后计算分布概率和熵
	- 熵越高证明输出差异越大，需要提问

CLAM: Selective Clarification for Ambiguous Questions with Generative Language Models
 - 提出CLAM框架：
	- 接收问题，判断问题是否模糊，通过计算下一个token为true的对数概率作为连续的预测分数
	- 生成澄清问题
	- 获取澄清信息
	- 生成答案
 - 数据集:Ambiguous TriviaQA

Tree of Clarifications: Answering Ambiguous Questions with Retrieval-Augmented Large Language Models(EMNLP 2023)
 - 将问题澄清过程显示建模为澄清树
 - 根节点为原始问题，中间节点为在某一语义维度上澄清一次后的解释，边是在外部文档下的转换
 - 使用BFS扩展树，每层优先探索新的歧义维度
 - 基于LLM进行自验证剪枝，判断当前答案是否可以视为原问题的合理回答，false则剪枝
 - 构建完成后聚合未被剪枝节点输出最终answer

Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models(NeurIPS 2024)
 - 将熵和信息增益引入LLM规划过程
 - 构建搜索树，在树的每个节点计算期望信息增益
 - 计算累计奖励
### Optimization

ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions(NeurIPS 2025)
 - LLM被动交互，感知环境受限，缺乏用户个性化
 - 将上下文感知形式化，便于评分
 - 数据集：构建ContextAgentBench，用于评估“上下文感知主动 Agent”
 - 评估指标：
	 - 主动预测：Acc-P(预测“需要/不需要服务”的分类准确率),MD(漏检率),FD(误检率),RMSE(预测的主动性评分（1-5分）与真实评分之间的均方根误差)
	 - 工具调用：Precision,Recall,F1-score,Acc-Args(参数准确率,只有参数全对才计分)

PsyAdvisor: A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conversations(ACL 2025)
 - 心理领域的垂类应用，让Agent主动提问
 - 数据集构建：ProPsyC
	- 使用CoT框架，结合RAG进行半自动标注
	- 分类策略，记录策略原理和反应归因
 - 微调策略规划模型
 - 评估指标：
	- 稳定性指标：R-1,R-L,B-2,B-4,Fbert 
	- 增强性指标：SFR(策略适配率，策略引发正向反应为适配，衡量模型提问时机)，PQR(主动提问率)
	- 人工评估
### Benchmark

Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents
 - 目前的智能体缺乏主动性，现有benchmark缺乏跨文档长时程，发现隐性问题能力的评估
 - 提出了 PROBE 框架，将流程解构为搜索，识别，解决三个模块
 - 数据集构建：基于用户画像数据集构建世界模型，生成瓶颈，增加干扰项构建用户存储，构建任务执行空间
 - 评估指标：
	 - 搜索：Precision,Recall,F1-score
	 - 识别：LLM-as-a-judge，评估关键细节和非关键细节，分数为(0,0.5,1)分
	 - 执行：规则匹配 + LLM-as-a-judge，分数为(0,0.5,1)

ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents
 - 主动式Agent的评估碎片化，缺乏标准
 - 提出ProactiveEval评估框架，把主动对话解耦为规划和引导两部分
 - 利用Human-AI Collaboration构建测试环境：
	 - 构建分层主题树以确保场景多样性
	 - LLM生成多个候选目标后用专家模型集成最优目标作为Reference
	 - Refinement：模糊化重写，注入噪声
 - Evalution：LLM-as-a-Judge
 - 评估指标：
	- 规划：打分
	- 引导：在五个维度上打分：有效性，个性化，语气，互动性，自然度
	- 补充分析：目标密度(每条消息中包含的子目标数量)
 - 推理模型急于在第一轮抛出所有计划，强化推理能力也许损害了对话交互感

UserBench: An Interactive Gym Environment for User-Centric Agents(NeurIPS 2025)
 - Agent benchmark忽略了真实交互中常常目标不完整，逐步显现，间接表达
 - 关注agent是否能主动澄清目标，识别隐含偏好，在偏好不完全，持续变化情况下做出决策
 - 设计了Userbench，一个可交互的gym环境+场景数据集合
 - 评估指标：
	- 最终决策质量：Normalized Score, Best Exist Rate, Correct Exist Rate
	- 交互与行为质量：Valid Search Attempt, Valid Action Attempt
	- 核心指标：Preference Elicited(挖出了用户多少偏好),Weighted Timing Score(时序惩罚，早期发现正确选项权重高，后期猜对惩罚)

Tell Me More! Towards Implicit User Intention Understanding of Language Model Driven Agents(ACL 2024)
- 用户指令具有模糊性，缺乏交互机制
- 提出IN3 Benchmark，评估Agent识别模糊指令和询问获取信息能力
- 设计了一个判断-思考提问-总结-提交给下游Agent的模块

### 数据集

 - 主动性：AR-Bench，ProactiveBench，UserBench，ContextAgentBench，ConsistentChat，Review-Instruct Multi-turn Dataset，PROBE
 - 特定任务：TravelPlanner，CollabLLM，DuRecDial
 - baseline数据集：Alpaca Dataset,ShareGPT,ChatAIpaca,UltraChat,Lmsys-Chat-1M,TopDial,Abg-CoQA,LIGHT
 - 行为分析：ProPsyC,Multiparty Chat Corpus,Think-aloud Study Data,PersonaChat
 - 综合评估：MMLU,MMLU-Pro,HumanEval,GSM8K,MATH,MT-Bench,Auto-Arena,MT-EVAL


### Innovation

- 将澄清机制建模为RL过程，State=对话历史，Action=是否问，问什么，Reward=信息增益(或仿照MC)Review-Instruct也可参照此方法改，探索最优学习机制
- RL的最终奖励如何合理分配到多轮澄清机制的每一步，建模为POMDP
- 多轮对话的反思，review，澄清机制是如何伤害性能的，在什么情况下会伤害
- 如何评估查询失败等情况是如何发生的？建模failure trajectory？
- 建模澄清机制的信息增益等，将澄清建模成决策(UOT那篇好像已经做了)
- 针对用户可能变化的想法，维护用户意图的动态贝叶斯置信分布
- 加CRC框架
- 在收到问题时进行plan，规划问答路线

### 总结
目前多轮对话中的用户意图分析与主动推理方面的工作基本集中于以下方面：
1. 基于不确定性消除的主动澄清机制
   - Ask-before-Plan提出了CEP框架，设计了澄清Agent，先判断是否模糊，模糊则进行澄清
   - Ask-before-Plan还使用了执行Agent进行反馈，帮助澄清Agent修改
   - CollabLLM使用类MC方法的RL直接优化对话，提高主动澄清质量
   - From Passive to Active Reasoning引入了信息覆盖率。模型计算当前信息是否足以支撑唯一解来量化不确定性，决定继续提问或给出结论
2. 基于环境的意图预测
	- Proactive Agent观察事件流预测用户意图计算收益，决定是否帮助用户
	- ContextAgent把环境信息形式化为感知，通过上下文感知确定用户需求
3. 模拟内在思维
	- Proactive Conversational Agents with Inner Thoughts 维护一个记忆库，通过显著性评分检索记忆生成想法，想法评估分数超过阈值才回复
	- ConsistentChat在生成对话前先建模意图，生成全局骨架保证对话一致性
4. 挖掘隐性偏好
	- UserBench要求通过交互行为找出用户隐性偏好
	- Beyond Reactivity的识别阶段要求结合用户画像和世界模型主动发现用户未提及的干扰项
5. 反思与意图校准
	- CRC框架强制模型在回复前进行一致性反思
	- ReviewInstruct在两轮对话间加入review阶段
	- Reward-Driven Interaction引入领域-意图分类作为辅助任务，强制模型在含噪数据中显式建模核心意图