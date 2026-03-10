# Corpus-Selection-Focused Literature for `webwalker`

这份清单只服务一个论文叙事：

`在自然超链接语料上，query-time corpus selection 能否比 dense top-k / iterative dense retrieval / eager GraphRAG indexing 更便宜、更准、更可控。`

完整结构化数据见 [corpus-selection-literature.tsv](/Users/acture/.codex/worktrees/898a/webwalker/docs/corpus-selection-literature.tsv)。

## Summary

- 主清单：32 篇
- Optional appendix：8 篇
- Argument support：8 篇
- 主清单 bucket：
  - `problem-framing-and-critiques`：6
  - `dense-and-path-selection-baselines`：10
  - `hyperlink-navigation-and-graph-search-adjacent`：6
  - `subgraph-selection-and-graph-retrieval-comparators`：6
  - `light-rag-and-graphrag-comparators`：4
- 叙事比例：
  - `problem + baseline` = 20 / 32
  - `QA+retrieval` 明显占主导
  - `hyperlink / graph search` 作为独特性支撑，而不是主战场

## 10 篇必读

1. [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/)
2. [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://aclanthology.org/2020.coling-main.580/)
3. [MuSiQue: Multihop Questions via Single-hop Question Composition](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996)
4. [Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://aclanthology.org/P19-1416/)
5. [Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://openreview.net/forum?id=SJgVHkrYDH)
6. [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://openreview.net/forum?id=EMHoBG0avc1)
7. [Answering Any-hop Open-domain Questions with Iterative Document Reranking](https://doi.org/10.1145/3404835.3462853)
8. [End-to-End Goal-Driven Web Navigation](https://papers.nips.cc/paper/6064-end-to-end-goal-driven-web-navigation)
9. [Focused crawling: A new approach to topic-specific Web resource discovery](https://doi.org/10.1016/S1389-1286(99)00052-3)
10. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html)

## 写 Related Work 的引用骨架

### 1. 先定义问题，不要先讲系统

- 用 `HotpotQA`、`2WikiMultiHopQA`、`MuSiQue` 定义 multi-hop setting。
- 用 `Compositional Questions Do Not Necessitate Multi-hop Reasoning` 说明只看最终答案会高估系统，必须关注证据与路径。
- 用 `IIRC` 说明“自然链接文档”不是边缘设定，而是一个明确存在的任务环境。
- 用 `Multi-Hop Question Answering` survey 给出任务版图，避免把你的工作误写成新的 answerer。

### 2. 把主战场钉在 corpus selection，而不是 answer generation

- 用 `Multi-Hop Paragraph Retrieval`、`Bridge Reasoning`、`Learning to Retrieve Reasoning Paths`、`MDR`、`HopRetriever`、`Any-hop Iterative Document Reranking` 说明已有工作已经把问题往“选对文档/路径”上推进。
- 用 `Combining Lexical and Dense Retrieval`、`Dense Hierarchical Retrieval`、`Select, Answer and Explain`、`DecompRC` 说明还有混合检索、层次检索、解释型选择器、分解式选择器这些分支。
- 你的切入点不是“再做一个多跳 RAG”，而是“在自然超链接语料上，selection unit 改成路径诱导子图”。

### 3. 说明自然超链接为什么值得单独拿出来做

- 用 `End-to-End Goal-Driven Web Navigation` 说明“按目标在链接图上顺序选择节点”本身就是成熟问题。
- 用 `Focused crawling`、`Topic-Sensitive PageRank`、`Personalized PageRank` survey 说明 query-conditioned graph prior 和 frontier control 是经典工具。
- 用 `Overcoming low-utility facets`、`ontology-based focused crawling` 说明表面 lexical match 不够，需要更结构化的 link-context / concept signal。

### 4. 再把输出形式从“文档集”推进到“路径诱导子图”

- 用 `HippoRAG`、`KG2RAG`、`PathRAG`、`Walk&Retrieve`、`SG-RAG MOT`、`SentGraph` 说明图检索和子图组织已经被证明对复杂问题有价值。
- 你的区别不在“也用图”，而在“图来自自然超链接语料，不需要 eager graph construction”。
- `GraphRAG` survey、`RAPTOR`、`LightRAG`、`When to use Graphs in RAG` 只作为次级对照，用来界定你不是在重做全链路 GraphRAG。

## 实验部分最该对比的 baseline

### A. 必做 selection baseline

- `BM25 top-k`：作为最朴素 lexical retrieval 上界/下界。
- `Dense top-k`：推荐 `DPR` / `Contriever` / `E5` 中选一个最稳定的双塔检索器。
- `Hybrid top-k`：对应 `Combining Lexical and Dense Retrieval`。
- `MDR`：你的核心强基线。
- `HopRetriever`：多跳 hop-wise retrieval 基线。
- `Answering Any-hop Open-domain Questions with Iterative Document Reranking`：迭代筛文档强基线。
- `Learning to Retrieve Reasoning Paths over Wikipedia Graph`：最直接的 path retrieval 对照。

### B. 必做自然链接 / 图结构 baseline

- `topic-sensitive / personalized PageRank`：只用图先验，不用语义 walker。
- `topology-only walk`：只用 hyperlink topology，不用 anchor/sentence context。
- `anchor-text-only scorer`：只用锚文本，不用周围句子。
- `link-context scorer`：只用 anchor + sentence，不用图先验。
- `HippoRAG`：图检索强基线。
- `KG2RAG` 或 `PathRAG`：子图/路径组织强基线。

### C. 次级对照，不要喧宾夺主

- `LightRAG`
- `RAPTOR`
- `IRCoT`
- `Adaptive-RAG`

这组可以做 secondary comparison，但不要把论文重心拖回 end-to-end RAG。

### D. 你自己的关键 ablation

- `no-link`：退化为纯 dense top-k
- `topology-only`：只看超链接图结构
- `link-context-only`：只看锚文本和周围句子
- `path-induced subgraph` vs `flat selected docs`
- `fixed budget` vs `adaptive budget`

## Additional Argument Support

这组不是你最核心的 baseline，也不建议全部放进主 Related Work 主体，但它们很适合在开题、答辩或 rebuttal 里支撑你的 argument。

### A. 支持“vanilla RAG / 单次检索对多跳问题不够”

- [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391)  
  直接给出一个 multi-hop RAG benchmark，并明确展示现有 RAG 检索与推理都不理想。
- [Optimizing Question Semantic Space for Dynamic Retrieval-Augmented Multi-hop Question Answering](https://arxiv.org/abs/2506.00491)  
  明确指出单轮 retrieve-and-read 容易遇到 semantic mismatch 与 subquestion dependency 成本。

### B. 支持“复杂问题更需要 evidence-driven selection，而不是堆更多上下文”

- [FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation](https://arxiv.org/abs/2510.22344)  
  用显式 evidence gap analysis 支撑“要找缺什么证据”，而不是盲目继续搜。
- [PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering](https://arxiv.org/abs/2510.14278)  
  强调 compact yet comprehensive evidence set，这和你的 corpus selection 目标高度一致。
- [Vendi-RAG: Adaptively Trading-Off Diversity And Quality Significantly Improves Retrieval Augmented Generation With LLMs](https://arxiv.org/abs/2502.11228)  
  支持“只看相关性不够，证据覆盖和多样性也重要”。
- [Replace, Don't Expand: Mitigating Context Dilution in Multi-Hop RAG via Fixed-Budget Evidence Assembly](https://arxiv.org/abs/2512.10787)  
  直接支撑你的预算论证：固定 budget 下优化 evidence assembly 比不断扩 context 更合理。
- [When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering](https://arxiv.org/abs/2601.19827)  
  很适合帮你反驳“把所有证据一次性交给模型就行了”，因为它说明 staged retrieval 本身可能更优。

### C. 支持“自然超链接图上的导航不是拍脑袋，而是有可学习结构”

- [Exploring Hierarchy-Aware Inverse Reinforcement Learning](https://arxiv.org/abs/1807.05037)  
  它对 Wikispeedia 玩家目标预测的结果说明，人类在 hyperlink graph 上确实表现出层次化规划结构。

## 主清单

### problem-framing-and-critiques

- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/)  
  `writing_role=problem | priority=P0 | thesis_use=定义问题`
- [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://aclanthology.org/2020.coling-main.580/)  
  `writing_role=problem | priority=P0 | thesis_use=定义问题`
- [MuSiQue: Multihop Questions via Single-hop Question Composition](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996)  
  `writing_role=problem | priority=P0 | thesis_use=定义问题`
- [Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://aclanthology.org/P19-1416/)  
  `writing_role=problem | priority=P0 | thesis_use=定义问题`
- [IIRC: A Dataset of Incomplete Information Reading Comprehension Questions](https://aclanthology.org/2020.emnlp-main.86/)  
  `writing_role=problem | priority=P1 | thesis_use=证明自然链接价值`
- [Multi-Hop Question Answering](https://arxiv.org/abs/2204.09140)  
  `writing_role=problem | priority=P1 | thesis_use=定义问题`

### dense-and-path-selection-baselines

- [Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://openreview.net/forum?id=SJgVHkrYDH)  
  `writing_role=baseline | priority=P0 | thesis_use=设定基线`
- [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://openreview.net/forum?id=EMHoBG0avc1)  
  `writing_role=baseline | priority=P0 | thesis_use=设定基线`
- [HopRetriever: Retrieve Hops over Wikipedia to Answer Complex Questions](https://doi.org/10.1609/AAAI.V35I15.17568)  
  `writing_role=baseline | priority=P1 | thesis_use=设定基线`
- [Answering Any-hop Open-domain Questions with Iterative Document Reranking](https://doi.org/10.1145/3404835.3462853)  
  `writing_role=baseline | priority=P0 | thesis_use=设定基线`
- [Simple yet Effective Bridge Reasoning for Open-Domain Multi-Hop Question Answering](https://aclanthology.org/D19-5806/)  
  `writing_role=baseline | priority=P1 | thesis_use=设定基线`
- [Multi-Hop Paragraph Retrieval for Open-Domain Question Answering](https://aclanthology.org/P19-1222/)  
  `writing_role=baseline | priority=P1 | thesis_use=设定基线`
- [Combining Lexical and Dense Retrieval for Computationally Efficient Multi-hop Question Answering](https://aclanthology.org/2021.sustainlp-1.7/)  
  `writing_role=baseline | priority=P2 | thesis_use=设定基线`
- [Dense Hierarchical Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.findings-emnlp.19/)  
  `writing_role=baseline | priority=P2 | thesis_use=设定基线`
- [Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents](https://doi.org/10.1609/aaai.v34i05.6441)  
  `writing_role=baseline | priority=P2 | thesis_use=设定基线`
- [Multi-hop Reading Comprehension through Question Decomposition and Rescoring](https://aclanthology.org/P19-1613/)  
  `writing_role=baseline | priority=P2 | thesis_use=设定基线`

### hyperlink-navigation-and-graph-search-adjacent

- [End-to-End Goal-Driven Web Navigation](https://papers.nips.cc/paper/6064-end-to-end-goal-driven-web-navigation)  
  `writing_role=adjacent | priority=P0 | thesis_use=证明自然链接价值`
- [Focused crawling: A new approach to topic-specific Web resource discovery](https://doi.org/10.1016/S1389-1286(99)00052-3)  
  `writing_role=adjacent | priority=P1 | thesis_use=证明自然链接价值`
- [Topic-Sensitive PageRank](https://doi.org/10.1145/511446.511513)  
  `writing_role=adjacent | priority=P1 | thesis_use=证明自然链接价值`
- [A Survey on Personalized PageRank Computation Algorithms](https://doi.org/10.1109/ACCESS.2019.2952653)  
  `writing_role=adjacent | priority=P2 | thesis_use=证明自然链接价值`
- [Overcoming low-utility facets for complex answer retrieval](https://doi.org/10.1007/s10791-018-9343-0)  
  `writing_role=adjacent | priority=P2 | thesis_use=证明自然链接价值`
- [An ontology-based approach to learnable focused crawling](https://doi.org/10.1016/j.ins.2008.07.030)  
  `writing_role=adjacent | priority=P2 | thesis_use=证明自然链接价值`

### subgraph-selection-and-graph-retrieval-comparators

- [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html)  
  `writing_role=baseline | priority=P0 | thesis_use=设定基线`
- [Knowledge Graph-Guided Retrieval Augmented Generation](https://arxiv.org/abs/2502.06864)  
  `writing_role=baseline | priority=P0 | thesis_use=支撑路径/子图输出`
- [PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths](https://arxiv.org/abs/2502.14902)  
  `writing_role=baseline | priority=P1 | thesis_use=支撑路径/子图输出`
- [Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks](https://arxiv.org/abs/2505.16849)  
  `writing_role=baseline | priority=P1 | thesis_use=设定基线`
- [SG-RAG MOT: SubGraph Retrieval Augmented Generation with Merging and Ordering Triplets for Knowledge Graph Multi-Hop Question Answering](https://doi.org/10.3390/make7030074)  
  `writing_role=comparator | priority=P2 | thesis_use=支撑路径/子图输出`
- [SentGraph: Hierarchical Sentence Graph for Multi-hop Retrieval-Augmented Question Answering](https://arxiv.org/abs/2601.03014)  
  `writing_role=comparator | priority=P2 | thesis_use=支撑路径/子图输出`

### light-rag-and-graphrag-comparators

- [Retrieval-Augmented Generation with Graphs (GraphRAG)](https://arxiv.org/abs/2501.00309)  
  `writing_role=comparator | priority=P1 | thesis_use=作为次级对照`
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://openreview.net/forum?id=GN921JHCRw)  
  `writing_role=comparator | priority=P1 | thesis_use=作为次级对照`
- [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-emnlp.568/)  
  `writing_role=comparator | priority=P1 | thesis_use=作为次级对照`
- [When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2506.05690)  
  `writing_role=comparator | priority=P1 | thesis_use=作为次级对照`

## Optional Appendix

### agentic / iterative RAG

- [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/)
- [Tree of Reviews: A Tree-based Dynamic Iterative Retrieval Framework for Multi-hop Question Answering](https://arxiv.org/abs/2404.14464)
- [Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering](https://arxiv.org/abs/2406.14891)
- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://aclanthology.org/2024.naacl-long.389/)

### stronger graph-RAG variants

- [D-RAG: Differentiable Retrieval-Augmented Generation for Knowledge Graph Question Answering](https://aclanthology.org/2025.emnlp-main.1793/)
- [StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization](https://arxiv.org/abs/2410.08815)

### focused crawling / RL adjacent

- [Tree-based Focused Web Crawling with Reinforcement Learning](https://arxiv.org/abs/2112.07620)
- [Weakly supervised learning for an effective focused web crawler](https://doi.org/10.1016/j.engappai.2024.107944)

## Argument Support Appendix

- [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391)
- [FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation](https://arxiv.org/abs/2510.22344)
- [PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering](https://arxiv.org/abs/2510.14278)
- [Optimizing Question Semantic Space for Dynamic Retrieval-Augmented Multi-hop Question Answering](https://arxiv.org/abs/2506.00491)
- [Vendi-RAG: Adaptively Trading-Off Diversity And Quality Significantly Improves Retrieval Augmented Generation With LLMs](https://arxiv.org/abs/2502.11228)
- [Replace, Don't Expand: Mitigating Context Dilution in Multi-Hop RAG via Fixed-Budget Evidence Assembly](https://arxiv.org/abs/2512.10787)
- [When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering](https://arxiv.org/abs/2601.19827)
- [Exploring Hierarchy-Aware Inverse Reinforcement Learning](https://arxiv.org/abs/1807.05037)
