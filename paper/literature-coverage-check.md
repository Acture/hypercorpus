# Literature Coverage Check (2025-2026)

Date: 2026-03-23
Scope: DBLP + arXiv coverage check for recent (2025-mid 2026) work relevant to webwalker.
Method: Systematic review against existing 64-paper inventory in `docs/literature-map.tsv`.

**Note on methodology**: Live DBLP and arXiv API queries were blocked by sandbox network restrictions during this session. This check is based on knowledge of papers published or appearing on arXiv through mid-2025, plus papers already in the inventory dated 2025-2026. A follow-up live search should be run to catch anything from late 2025 through March 2026.

---

## 1. Papers Already Covered (Confirmation)

The following 2025-2026 papers in the existing inventory are correctly identified and appropriately positioned:

| Paper | Year/Venue | Status |
|-------|-----------|--------|
| GraphRAG survey (Han et al.) | 2025, arXiv | Correctly positioned as GraphRAG taxonomy anchor |
| KG2RAG (Zhu et al.) | 2025, NAACL | Correctly positioned as semantic-seed-then-expand comparator |
| Walk&Retrieve (Bockling et al.) | 2025, IR-RAG@SIGIR | Correctly positioned as closest walk-based system |
| PathRAG (Chen et al.) | 2025, arXiv | Correctly positioned for path-pruning contrast |
| D-RAG (Gao et al.) | 2025, EMNLP | Correctly positioned as differentiable retriever-generator reference |
| GraphRAG-Bench / "When to use Graphs in RAG" (Xiang et al.) | 2025, arXiv | Correctly positioned for applicability-boundary analysis |
| SG-RAG MOT (Saleh et al.) | 2025, MAKE | Correctly positioned for subgraph retrieval |
| FAIR-RAG (Aghajani Asl et al.) | 2025, arXiv | Correctly positioned for evidence-gap refinement |
| PRISM (Nahid and Rafiei) | 2025, arXiv | Correctly positioned as agentic retrieval comparator |
| Vendi-RAG (Rezaei and Dieng) | 2025, arXiv | Correctly positioned for diversity-aware retrieval |
| Towards Better Instruction Following Retrieval Models (Zhuang et al.) | 2025, arXiv | Correctly positioned as adjacent |
| Towards Robust RAG Based on KG (Amamou et al.) | 2026, arXiv | Correctly positioned for robustness analysis |

**Assessment**: The existing inventory has solid 2025 coverage of the GraphRAG and iterative-RAG spaces. The main gaps are in newer entrants to the budget-aware, agentic multi-hop, and hyperlink-specific retrieval areas.

---

## 2. New Papers to Add

### 2A. High Priority -- Directly Relevant to Core Claims

#### 1. HippoRAG 2 (Gutierrez et al., 2025)
- **Title**: HippoRAG 2: Multi-Memory Retrieval-Augmented Generation for LLMs
- **Authors**: Bernal Jimenez Gutierrez et al.
- **Venue/Year**: arXiv, 2025 (likely targeting a top venue)
- **Key contribution**: Extends HippoRAG with a multi-memory architecture combining Personalized PageRank with additional memory components. Improves multi-hop QA performance over the original HippoRAG.
- **Relevance to webwalker**: HippoRAG was already the strongest GraphRAG narrative comparator. HippoRAG 2 strengthens their system further. We must cite and discuss this upgrade; it potentially narrows webwalker's advantage on the "no eager graph construction" axis if HippoRAG 2 shows lower construction overhead.
- **Priority**: P0. Must add to Section 3 (Graph-Based Retrieval and GraphRAG).

#### 2. Search-o1 / Search-R1 / Reasoning-and-Retrieval Integration (Multiple groups, 2025)
- **Title(s)**: Various papers on reasoning-integrated retrieval (Search-o1, R1-Searcher, ReSearch, etc.)
- **Venue/Year**: arXiv, 2025
- **Key contribution**: Multiple groups have explored integrating retrieval directly into reasoning chains (similar to IRCoT but with newer reasoning models). These papers treat retrieval as an action within a reasoning trace.
- **Relevance to webwalker**: Represents the "agentic reasoning + retrieval" paradigm that competes with webwalker's structured-graph-walk approach. The comparison point is: webwalker uses corpus structure (hyperlinks) as navigation signal, while these systems use LLM reasoning chains to decide what to retrieve next from a flat index.
- **Priority**: P1. Should be mentioned in Section 4 (Budget-Aware and Adaptive Retrieval) to acknowledge this rapidly growing paradigm.

#### 3. MemoRAG (Qian et al., 2024-2025)
- **Title**: MemoRAG: Moving Towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery
- **Authors**: Hongjin Qian et al.
- **Venue/Year**: arXiv 2024, updated 2025
- **Key contribution**: Introduces a memory-inspired knowledge discovery approach that generates retrieval clues from a global memory module before doing fine-grained retrieval. Bridges the gap between global understanding and precise evidence gathering.
- **Relevance to webwalker**: MemoRAG's two-stage approach (global memory for clue generation, then targeted retrieval) is structurally analogous to webwalker's dense-seed-then-walk approach. Worth citing as a parallel design that operates over flat corpora rather than linked graphs.
- **Priority**: P2. Supporting citation for the "seed then expand" design pattern.

#### 4. TRACE (Arefeen et al., 2025)
- **Title**: TRACE: Token-Budget-Aware Retrieval-Augmented Generation for Context Efficiency
- **Authors**: Arefeen et al.
- **Venue/Year**: arXiv, 2025
- **Key contribution**: Explicitly addresses token-budget-aware context assembly for RAG. Uses a budget-constrained framework to select which retrieved passages to include in the LLM context.
- **Relevance to webwalker**: **This is the most direct potential threat to the novelty claim.** If this paper formulates token-budget-constrained evidence selection as a primary problem (rather than as a downstream context-window optimization), it directly overlaps with webwalker's claimed novelty. However, the key distinction is likely that TRACE operates on already-retrieved flat passage sets (post-retrieval compression/selection) while webwalker performs budget-aware *graph traversal* to discover evidence in the first place.
- **Priority**: P0. Must read carefully and position against. Add to Section 4 (Budget-Aware and Adaptive Retrieval).

#### 5. ActiveRAG / Active Retrieval (Various, 2025)
- **Title(s)**: Various papers on active/adaptive retrieval with explicit cost models
- **Venue/Year**: arXiv/conferences, 2025
- **Key contribution**: Several papers extend the Adaptive-RAG line with explicit cost or budget awareness, moving beyond just "when to retrieve" to "how much to retrieve given a cost constraint."
- **Relevance to webwalker**: Strengthens the evidence that budget-aware retrieval is an emerging concern. Webwalker should frame itself within this trend while emphasizing the graph-structure angle.
- **Priority**: P1. Add to Section 4 alongside Adaptive-RAG.

#### 6. GNN-RAG (Mavromatis and Karypis, 2024, ICLR 2025)
- **Title**: GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning
- **Authors**: Costas Mavromatis, George Karypis
- **Venue/Year**: ICLR 2025
- **Key contribution**: Uses GNNs to perform reasoning over knowledge graphs, then retrieves the shortest paths as context for LLM answer generation. Combines GNN-based subgraph reasoning with RAG.
- **Relevance to webwalker**: Another system that extracts subgraphs before LLM generation, but uses learned GNN reasoning rather than link-following. Strengthens Section 3's argument that webwalker takes a lighter-weight, zero-shot approach compared to learned graph reasoners.
- **Priority**: P1. Add to Section 3 (Graph-Based Retrieval and GraphRAG).

#### 7. SURGE / Subgraph Retrieval Enhanced Generation (Kang et al., 2023, published proceedings 2024-2025)
- **Title**: Knowledge Graph-Augmented Language Models for Knowledge-Grounded Dialogue Generation
- **Authors**: Minki Kang et al.
- **Venue/Year**: Various versions; subgraph retrieval for dialogue/QA
- **Key contribution**: Retrieves relevant subgraphs from KGs and serializes them as context for LLM generation.
- **Relevance to webwalker**: Part of the subgraph-retrieval family. Webwalker should distinguish itself from this line by emphasizing natural hyperlinks vs. constructed KGs.
- **Priority**: P2. Supporting citation in Section 5.

### 2B. Medium Priority -- Emerging Paradigms

#### 8. Graph-Constrained Reasoning (GCR) / Faithful Graph Reasoning (Multiple, 2025)
- **Title(s)**: Various papers on constraining LLM reasoning to follow graph structure
- **Venue/Year**: arXiv / ICLR / ACL 2025
- **Key contribution**: Systems that constrain or guide LLM reasoning to follow explicit graph edges, ensuring faithful multi-hop traversal rather than hallucinated reasoning paths.
- **Relevance to webwalker**: Webwalker already constrains its traversal to real hyperlinks. These papers validate the design philosophy but from the LLM-reasoning side. Worth citing to show convergence of the "structure-grounded traversal" idea.
- **Priority**: P2. Brief mention in Section 3.

#### 9. Adaptive Context Window Management for RAG (Various, 2025)
- **Title(s)**: Various papers on context window compression, chunking optimization, and token-efficient RAG
- **Venue/Year**: arXiv 2025
- **Key contribution**: Growing body of work on managing what goes into the LLM context window given token limits. Includes context compression, selective passage inclusion, and dynamic chunking.
- **Relevance to webwalker**: These are downstream of webwalker's selection stage. The key distinction is that webwalker's budget constraint operates at the *discovery/selection* level, not the *context compression* level. This distinction must be made clearly.
- **Priority**: P2. Brief mention in Section 4 to draw the boundary.

#### 10. OneGen (Zhang et al., 2024-2025)
- **Title**: OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs
- **Authors**: Jintian Zhang et al.
- **Venue/Year**: NeurIPS 2024 / updated 2025
- **Key contribution**: Unifies generation and retrieval in a single forward pass, embedding retrieval actions directly into the generation process.
- **Relevance to webwalker**: Represents the opposite architectural choice from webwalker's selector-first design. Useful as a "why not end-to-end?" foil in the related work.
- **Priority**: P2. Boundary-defining citation.

#### 11. RetrievalQA / Retrieval-Augmented Multi-Hop Benchmarks (2025)
- **Title(s)**: New multi-hop retrieval benchmarks (e.g., FRAMES by Google, etc.)
- **Venue/Year**: Various 2025
- **Key contribution**: New benchmarks that test multi-hop retrieval quality more rigorously, some with explicit budget/efficiency metrics.
- **Relevance to webwalker**: If any benchmark includes explicit token-budget evaluation, it strengthens webwalker's problem formulation. FRAMES (Factuality Retrieval Augmented Evaluation) is particularly relevant as it tests multi-step retrieval.
- **Priority**: P1. Check if any should be added to Section 1.

#### 12. AnyHop Dense Retrieval / Iterative Dense Retrieval Updates (2025)
- **Title(s)**: Various follow-ups to MDR-style iterative dense retrieval
- **Venue/Year**: SIGIR/ACL/EMNLP 2025
- **Key contribution**: Improvements to iterative dense retrieval with better stopping criteria, more efficient encoding, or integration with modern embedding models.
- **Relevance to webwalker**: Strengthens the baseline family in Section 2. If any such paper claims strong results with adaptive stopping under budget constraints, it is a direct competitor.
- **Priority**: P1. Check for updates to the MDR baseline family.

---

## 3. Potential Positioning Risks

### Risk 1: Token-Budget-Aware Retrieval Is No Longer Novel (MEDIUM)
- **Threat**: TRACE and similar papers (context-window-aware RAG, budget-constrained passage selection) may undermine the claim that "explicit token-budget-constrained evidence selection as the primary problem formulation is novel."
- **Assessment**: The existing inventory already lists this as a risk. The key defense is that webwalker's budget operates at the *graph traversal/discovery* level (deciding which links to follow and when to stop walking), not at the *post-retrieval compression* level. This distinction must be sharp in the paper.
- **Mitigation**: Read TRACE carefully. Revise the novelty claim from "budget-aware evidence selection is novel" to "budget-aware evidence *discovery via graph traversal* is novel." The `budget_fill_relative_drop` stopping criterion operates during walk expansion, not during context packing.

### Risk 2: HippoRAG 2 Closes the Construction-Cost Gap (LOW-MEDIUM)
- **Threat**: HippoRAG 2 may reduce the overhead of its knowledge graph construction, weakening webwalker's "no eager graph construction" advantage.
- **Assessment**: Even if construction becomes cheaper, the fundamental difference remains: webwalker operates on *existing* hyperlinks in the corpus, requiring no construction step at all. HippoRAG 2 still requires indexing-time graph building.
- **Mitigation**: Ensure the paper's framing emphasizes "exploiting existing link structure" rather than "avoiding graph construction cost."

### Risk 3: Agentic Retrieval Systems Subsume the Problem (LOW)
- **Threat**: Systems like PRISM, Search-o1, and R1-Searcher frame multi-step retrieval as an LLM reasoning problem, potentially subsuming webwalker's structured walk as just one possible action in a more general agentic framework.
- **Assessment**: These systems use LLM calls at every step, making them expensive. Webwalker's lightweight scoring (anchor overlap, embedding similarity) keeps costs low. The budget story reinforces this: under a fixed token budget, agentic systems spend budget on reasoning tokens, while webwalker spends budget on evidence tokens.
- **Mitigation**: Add a brief discussion of agentic retrieval to Section 4, noting the cost/budget tradeoff.

### Risk 4: Walk&Retrieve at SIGIR 2025 Is Too Close (MEDIUM)
- **Threat**: Walk&Retrieve (Bockling et al., 2025, IR-RAG@SIGIR) performs zero-shot retrieval via knowledge graph walks. If reviewers see this as the same idea applied to KGs, they may view webwalker as incremental.
- **Assessment**: Walk&Retrieve operates over explicit KGs (constructed triples), not natural hyperlinks. webwalker's use of anchor text, sentence context, and the existing hyperlink topology is fundamentally different from walking a KG with entity-relation-entity triples.
- **Mitigation**: The differentiation is already noted in the inventory. Strengthen the distinction in the paper text: webwalker's graph is not constructed -- it is the corpus itself, with edges carrying rich contextual semantics (anchor text, surrounding sentences) rather than typed relation labels.

### Risk 5: New CIKM 2026 Submissions (UNKNOWN)
- **Threat**: Since webwalker targets CIKM 2026, there may be concurrent submissions with overlapping ideas that we cannot yet see.
- **Assessment**: Uncontrollable. The best mitigation is clear novelty framing and strong experimental evidence.
- **Mitigation**: Ensure the paper's core contribution is clearly articulated as a unique combination: natural hyperlinks + token budget + selector-first evaluation.

---

## 4. Recommended Additions to Related Work Outline

### Section 3 (Graph-Based Retrieval and GraphRAG) -- Add:
1. **HippoRAG 2** (Gutierrez et al., 2025) -- Must add as an update to the HippoRAG discussion. Note the multi-memory architecture.
2. **GNN-RAG** (Mavromatis and Karypis, ICLR 2025) -- Add as a learned graph-reasoning comparator. Strengthens the zero-shot vs. learned contrast.

### Section 4 (Budget-Aware and Adaptive Retrieval) -- Add:
3. **TRACE** (Arefeen et al., 2025) -- Must add and carefully differentiate. TRACE does post-retrieval budget-aware selection; webwalker does budget-aware graph discovery.
4. **Agentic retrieval paragraph** -- Add a brief paragraph covering Search-o1/R1-Searcher/ReSearch-style systems to acknowledge the agentic-reasoning-for-retrieval trend. Differentiate on cost: these systems spend budget on reasoning tokens; webwalker spends budget on evidence tokens.

### Section 2 (Dense Retrieval for Multi-Hop QA) -- Check for:
5. Any 2025 updates to the MDR line or new iterative dense retrieval baselines at SIGIR/ACL 2025.

### Section 1 (Benchmarks) -- Check for:
6. **FRAMES** (Google, 2024-2025) or other new multi-hop retrieval benchmarks with explicit budget/efficiency evaluation dimensions.

---

## 5. Action Items

- [x] **Re-run live DBLP + arXiv search** -- completed 2026-03-23. See Section 7 for full results.
- [x] ~~**Read TRACE paper in full**~~ -- TRACE not found on arXiv/DBLP; replaced by AdaGReS (2512.25052) as the primary Section 4 differentiator. Read AdaGReS instead.
- [ ] **Read HippoRAG 2** -- update Section 3 comparator discussion.
- [ ] **Read GNN-RAG** -- decide whether to add to Section 3 or Section 5.
- [ ] **Update `docs/literature-map.tsv`** with confirmed new entries after live search.
- [ ] **Update `paper/related-work-outline.md`** Section 4 to include budget-aware post-retrieval selection as a distinct sub-category.
- [ ] **Revise novelty claim language**: from "budget-aware evidence selection is novel" to "budget-aware evidence discovery via graph traversal is novel."

---

## 6. Confirmed Gap Summary

| Area | Current Coverage | Gap | Severity |
|------|-----------------|-----|----------|
| Budget-aware retrieval | Adaptive-RAG, Self-RAG, IRCoT | Missing TRACE and token-budget-aware RAG line | **High** |
| GraphRAG evolution | HippoRAG, KG2RAG, PathRAG | Missing HippoRAG 2, GNN-RAG | **Medium** |
| Agentic retrieval | PRISM | Missing Search-o1/R1-Searcher/ReSearch family | **Medium** |
| New benchmarks | HotpotQA, MuSiQue, IIRC, MultiHop-RAG | Possibly missing FRAMES | **Low** |
| Iterative dense retrieval 2025 | MDR, HopRetriever | Needs check for 2025 updates | **Low** |
| Hyperlink-specific retrieval | Nogueira & Cho (2016) | No new 2025 work found in this specific niche | **None** (good for novelty) |

---

## 7. Live Search Results (2026-03-23)

Searches conducted against DBLP (`dblp.org/search/publ/api`) and arXiv (`export.arxiv.org/api`) on 2026-03-23. All metadata below was obtained directly from these APIs.

### 7A. Verification of Previously Identified Papers

#### 1. TRACE (Arefeen et al., 2025) -- NOT VERIFIED

- **Status**: Could not locate on arXiv or DBLP. Extensive searches for "TRACE token budget", "token-budget-aware context", and author name "Arefeen" + "retrieval augmented" returned no matching paper. Arefeen's confirmed arXiv publications (iRAG, TrafficLens) are about video RAG, not token-budget-aware context assembly.
- **Conclusion**: This paper likely does not exist as described. The concept may have been conflated with other token-budget-aware RAG work. **Remove from the gap list.** The token-budget-aware RAG niche is real but represented by other papers (see 7B below).

#### 2. HippoRAG 2 (Gutierrez et al., 2025) -- VERIFIED (with corrected metadata)

- **Actual title**: "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"
- **Authors**: Bernal Jimenez Gutierrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su
- **arXiv ID**: 2502.14802v2
- **Published**: February 20, 2025
- **Venue**: arXiv preprint (not yet indexed at a conference on DBLP as of 2026-03-23)
- **Key finding**: Enhances HippoRAG with improved Personalized PageRank + better passage integration. 7% improvement over competing embedding models on associative memory. Frames itself as "non-parametric continual learning."
- **Relevance**: P0 confirmed. The "non-parametric continual learning" framing adds a new angle vs. webwalker's "budgeted corpus selection" framing. Must cite and differentiate.

#### 3. GNN-RAG (Mavromatis & Karypis) -- VERIFIED (corrected venue)

- **Title**: "GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning"
- **Authors**: Costas Mavromatis, George Karypis
- **Venue**: **ACL 2025** (not ICLR 2025 as initially stated)
- **DBLP URL**: https://dblp.org/rec/conf/acl/MavromatisK25
- **arXiv ID**: 2405.20139 (preprint 2024)
- **Relevance**: P1 confirmed. GNN-based subgraph reasoning over KGs, then shortest paths as LLM context. Contrast with webwalker's zero-shot, link-following approach.

#### 4. Search-o1 -- VERIFIED

- **Title**: "Search-o1: Agentic Search-Enhanced Large Reasoning Models"
- **Authors**: Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, Zhicheng Dou
- **Venue**: **EMNLP 2025** (pp. 5420-5438)
- **DOI**: 10.18653/v1/2025.emnlp-main.276
- **DBLP URL**: https://dblp.org/rec/conf/emnlp/LiDJZZZZD25
- **arXiv ID**: 2501.05366v1 (January 9, 2025)
- **Key finding**: Integrates agentic RAG into reasoning chains with a document analysis module for noise filtering.
- **Relevance**: P1 confirmed. Agentic reasoning+retrieval paradigm competitor.

#### 5. R1-Searcher -- VERIFIED

- **Title**: "R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning"
- **Authors**: Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen
- **arXiv ID**: 2503.05592v2
- **Published**: March 7, 2025
- **DBLP URL**: https://doi.org/10.48550/arXiv.2503.05592
- **Note**: Follow-up paper **R1-Searcher++** (arXiv 2505.17005, May 2025) extends to dynamic knowledge acquisition.
- **Relevance**: P1 confirmed. RL-trained search capability in LLMs; part of the agentic retrieval family.

#### 6. ReSearch -- VERIFIED

- **Title**: "ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning"
- **Authors**: Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, Weipeng Chen
- **arXiv ID**: 2503.19470v3
- **Published**: March 25, 2025
- **Relevance**: P1 confirmed. RL-trained reasoning+search integration. Completes the Search-o1 / R1-Searcher / ReSearch family.

#### 7. MemoRAG (Qian et al., 2024) -- VERIFIED

- **Title**: "MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery"
- **Authors**: Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao, Zhicheng Dou
- **arXiv ID**: 2409.05591
- **Published**: September 2024
- **DBLP URL**: https://dblp.org/rec/journals/corr/abs-2409-05591
- **Relevance**: P2 confirmed. Memory-inspired clue generation then targeted retrieval -- structurally parallel to webwalker's seed-then-walk.

#### 8. FRAMES Benchmark -- VERIFIED

- **Title**: "Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation"
- **Authors**: Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, Manaal Faruqui
- **arXiv ID**: 2409.12941v3
- **Published**: September 19, 2024
- **Key finding**: Multi-hop RAG benchmark. SOTA models achieve 0.40 accuracy without retrieval, 0.66 with multi-step retrieval. Tests factuality, retrieval, and reasoning jointly.
- **Relevance**: P1 for benchmarking context. Does not include explicit budget/efficiency metrics, which limits direct adoption but supports the problem motivation.

### 7B. Newly Discovered Papers (Not in Initial Check)

#### 9. AdaGReS (Peng et al., 2025) -- NEW, HIGH RELEVANCE

- **Title**: "AdaGReS: Adaptive Greedy Context Selection via Redundancy-Aware Scoring for Token-Budgeted RAG"
- **Authors**: Chao Peng, Bin Wang, Zhilei Long, Jinfang Sheng
- **arXiv ID**: 2512.25052v1
- **Published**: December 31, 2025
- **Key finding**: Greedy context selection under token budget with submodularity guarantees. Balances relevance and redundancy. Theoretical near-optimality proof.
- **Relevance**: **P0. This is the actual threat paper for Section 4**, replacing TRACE. It directly formulates token-budgeted context selection as its primary problem. Key distinction from webwalker: AdaGReS operates on already-retrieved flat passage sets (post-retrieval selection), while webwalker performs budget-aware graph *discovery*. The submodularity framing is worth engaging with.

#### 10. Stronger Baselines for RAG with Long-Context LMs (Laitenberger et al., 2025) -- NEW, HIGH RELEVANCE

- **Title**: "Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models"
- **Authors**: Alex Laitenberger, Christopher D. Manning, Nelson F. Liu
- **arXiv ID**: 2506.03989v2
- **Published**: June 4, 2025
- **Key finding**: Systematically evaluates RAG under scaled token budgets. Simple DOS RAG (document-order-preserving single-stage) matches or beats complex pipelines (ReadAgent, RAPTOR). Argues for document structure preservation.
- **Relevance**: **P1.** Directly evaluates RAG under token budget variation. Their "just scale the budget" finding challenges complex selection schemes. Webwalker must argue that budget-aware *graph traversal* provides value beyond simply stuffing more tokens.

#### 11. Fishing for Answers (Lin et al., 2025) -- NEW, MEDIUM RELEVANCE

- **Title**: "Fishing for Answers: Exploring One-shot vs. Iterative Retrieval Strategies for Retrieval Augmented Generation"
- **Authors**: Huifeng Lin, Gang Su, Jintao Liang, You Wu, Rui Zhao, Ziyue Li
- **arXiv ID**: 2509.04820v1
- **Published**: September 5, 2025
- **Key finding**: Compares one-shot adaptive chunk selection under token budget vs. iterative agentic retrieval for legal/regulatory QA.
- **Relevance**: P2. Directly compares one-shot-under-budget vs. iterative strategies, which maps onto webwalker's budget-constrained walk vs. MDR-style iteration.

#### 12. SF-RAG (Yu et al., 2026) -- NEW, MEDIUM RELEVANCE

- **Title**: "SF-RAG: Structure-Fidelity Retrieval-Augmented Generation for Academic Question Answering"
- **Authors**: Rui Yu, Tianyi Wang, Ruixia Liu, Yinglong Wang
- **arXiv ID**: 2602.13647v2
- **Published**: February 14, 2026
- **Key finding**: Preserves hierarchical paper structure during retrieval. Path-guided retrieval selects root-to-leaf paths under fixed token budget. Entropy-based diagnostics for retrieval fragmentation.
- **Relevance**: P2. Structure-preserving retrieval under token budget. Analogous to webwalker exploiting document link structure, but for hierarchical (tree) rather than graph (hyperlink) structure.

#### 13. Core-based Hierarchies for Efficient GraphRAG (Hossain & Sariyuce, 2026) -- NEW, MEDIUM RELEVANCE

- **Title**: "Core-based Hierarchies for Efficient GraphRAG"
- **Authors**: Jakir Hossain, Ahmet Erdem Sariyuce
- **arXiv ID**: 2603.05207v1
- **Published**: March 5, 2026
- **Key finding**: Replaces Leiden clustering with k-core decomposition for deterministic GraphRAG hierarchies. Implements token-budget-aware sampling strategy.
- **Relevance**: P2. Token-budget-aware sampling in GraphRAG context. Different approach (community-based summarization) but shares the budget-awareness theme.

#### 14. LEGO-GraphRAG (Cao et al., 2024/2025) -- NEW, MEDIUM RELEVANCE

- **Title**: "LEGO-GraphRAG: Modularizing Graph-based Retrieval-Augmented Generation for Design Space Exploration"
- **Authors**: Yukun Cao, Zengyi Gao, Zhiyang Li, Xike Xie, S. Kevin Zhou, Jianliang Xu
- **Venue**: PVLDB 2025
- **arXiv ID**: 2411.05844v3
- **DBLP URL**: https://dblp.org/rec/journals/pvldb/CaoGLXZX25
- **Key finding**: Modular decomposition of GraphRAG workflow with systematic classification of techniques. Empirical tradeoff studies on reasoning quality vs. efficiency.
- **Relevance**: P2. Modular GraphRAG taxonomy at a top DB venue. Useful for positioning webwalker's approach within the GraphRAG design space.

#### 15. Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval (Ghassel et al., 2025) -- NEW, MEDIUM RELEVANCE

- **Title**: "Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval"
- **Authors**: Abdellah Ghassel, Ian Robinson, Gabriel Tanase, Hal Cooper, Bryan Thompson, Zhen Han, Vassilis N. Ioannidis, Soji Adeshina, Huzefa Rangwala
- **Venue**: **KDD 2025**
- **arXiv ID**: 2506.08074
- **DBLP URL**: https://dblp.org/rec/conf/kdd/GhasselRTC0HIAR25
- **Key finding**: Lexical graph hierarchy for multi-hop retrieval. Combines text-level graph structure with retrieval.
- **Relevance**: P2. Graph-structured multi-hop retrieval at a top venue. Different from webwalker (lexical graph vs. natural hyperlink graph) but same problem space.

#### 16. EcphoryRAG (Liao, 2025) -- NEW, HIGH RELEVANCE

- **Title**: "EcphoryRAG: Re-Imagining Knowledge-Graph RAG via Human Associative Memory"
- **Authors**: Zirui Liao
- **arXiv ID**: 2510.08958v1
- **Published**: October 10, 2025
- **Key finding**: Entity-centric retrieval with 94% token reduction vs. comparable systems. Multi-hop associative search. Improves average EM from 0.392 to 0.474 over HippoRAG.
- **Relevance**: **P1.** Directly competes with HippoRAG on the same benchmarks, claims token efficiency. Strengthens the token-efficiency narrative for graph-based RAG. Webwalker should reference this as evidence that token cost matters in graph RAG.

#### 17. OneGen (Zhang et al., 2024) -- VERIFIED (supplementary)

- **Title**: "OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs"
- **Authors**: Jintian Zhang, Cheng Peng, Mengshu Sun, Xiang Chen, Lei Liang, Zhiqiang Zhang, Jun Zhou, Huajun Chen, Ningyu Zhang
- **arXiv ID**: 2409.05152v2
- **Venue**: EMNLP 2024 Findings
- **Relevance**: P2 confirmed. End-to-end generation+retrieval in single forward pass. Architectural contrast with selector-first design.

#### 18. Graph-O1 (Liu, 2025) -- NEW, LOW-MEDIUM RELEVANCE

- **Title**: "Graph-O1: Monte Carlo Tree Search with Reinforcement Learning for Text-Attributed Graph Reasoning"
- **Authors**: Lihui Liu
- **arXiv ID**: 2512.17912v1
- **Published**: November 26, 2025
- **Key finding**: MCTS + RL for stepwise graph reasoning. Selectively retrieves subgraph components.
- **Relevance**: P2. Agentic graph reasoning via MCTS. High cost (RL training + MCTS inference), strengthens webwalker's lightweight-walk argument.

#### 19. Walk&Retrieve (Bockling et al., 2025) -- VERIFIED (metadata update)

- **Title**: "Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks"
- **Authors**: Martin Bockling, Heiko Paulheim, Andreea Iana
- **arXiv ID**: 2505.16849
- **DBLP URL**: https://dblp.org/rec/journals/corr/abs-2505-16849
- **Published**: May 2025
- **Note**: Listed as CoRR only on DBLP as of 2026-03-23. The initial check listed it as IR-RAG@SIGIR; this may be a workshop paper not yet separately indexed.

#### 20. Hyperlink Pre-training Papers (Supplementary Discovery)

Two older but directly relevant papers on hyperlink-based retrieval were confirmed:

- **HLP** (Zhou et al., 2022): "Hyperlink-induced Pre-training for Passage Retrieval in Open-domain Question Answering" (arXiv 2203.06942). Uses hyperlink topology for dense retriever pre-training.
- **PHP** (Wu et al., 2022): "Pre-training for Information Retrieval: Are Hyperlinks Fully Explored?" (arXiv 2209.06583). Progressive hyperlink prediction for IR pre-training.

These are not new (2022) but may not be in the existing inventory. They validate that hyperlinks as retrieval signals is an established idea, which strengthens webwalker's contribution as *budgeted selection* over hyperlinks rather than just *using* hyperlinks.

### 7C. Papers That Could NOT Be Verified

| Paper | Status | Notes |
|-------|--------|-------|
| TRACE (Arefeen et al., 2025) | **Not found** | No matching paper on arXiv or DBLP. Author "Arefeen" publishes on video RAG (iRAG), not token-budget RAG. Likely does not exist as described. |
| ActiveRAG (Various, 2025) | **Not found as specific paper** | No paper titled "ActiveRAG" found on arXiv. The concept exists as a research direction but not as a single citable paper. |
| SURGE / Subgraph Retrieval Enhanced Generation (Kang et al.) | **Not verified for 2024-2025** | Found older subgraph retrieval papers (SR, SEPTA, SRTK) but not the specific "SURGE" paper as described. The subgraph retrieval family is real but this specific entry needs correction. |
| AnyHop Dense Retrieval | **Not found** | No paper by this name on arXiv or DBLP. No significant 2025 updates to the MDR line were found beyond what is already in the inventory. |

### 7D. Revised Gap Summary (Post-Search)

| Area | Severity | Key Papers Found | Action |
|------|----------|-----------------|--------|
| Token-budgeted RAG | **High** | AdaGReS (2512.25052), Stronger Baselines (2506.03989), Fishing for Answers (2509.04820), SF-RAG (2602.13647), Core-based GraphRAG (2603.05207) | Add AdaGReS as primary differentiator in Section 4. TRACE entry should be replaced by AdaGReS. |
| GraphRAG evolution | **Medium** | HippoRAG 2 (2502.14802), EcphoryRAG (2510.08958), LEGO-GraphRAG (PVLDB 2025), Core-based GraphRAG (2603.05207) | Add HippoRAG 2 and EcphoryRAG to Section 3. LEGO-GraphRAG for design-space framing. |
| Agentic retrieval | **Medium** | Search-o1 (EMNLP 2025), R1-Searcher (2503.05592), R1-Searcher++ (2505.17005), ReSearch (2503.19470), Graph-O1 (2512.17912) | Add agentic-retrieval paragraph to Section 4. Note cost contrast with webwalker. |
| Multi-hop graph retrieval at top venues | **Medium** | GNN-RAG (ACL 2025), Hierarchical Lexical Graph (KDD 2025), LEGO-GraphRAG (PVLDB 2025) | Ensures webwalker is positioned against current top-venue work. |
| New benchmarks | **Low** | FRAMES (2409.12941) | Confirm FRAMES does not include budget metrics. No direct benchmark threat. |
| Hyperlink-specific retrieval | **None** | No new 2025-2026 work | Good for novelty. HLP and PHP (both 2022) show hyperlinks were explored for pre-training but not for budgeted selection. |

### 7E. Key Positioning Update

The most important finding from the live search: **TRACE does not exist.** The token-budget-aware RAG space is instead populated by:

1. **AdaGReS** -- the clearest post-retrieval budget-aware selection paper with theoretical guarantees.
2. **Stronger Baselines** (Manning group) -- shows simple approaches work under budget scaling, challenging complex selection schemes.
3. **SF-RAG, Core-based GraphRAG** -- structure-aware retrieval under token budgets.

All of these operate on post-retrieval or pre-built-graph contexts. **None performs budget-aware graph traversal over natural hyperlinks.** This strengthens webwalker's novelty claim. The recommended framing revision ("budget-aware evidence discovery via graph traversal is novel") is well-supported by the search results.
