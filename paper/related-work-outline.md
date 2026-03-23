# Hypercorpus Related Work Outline

Purpose: paper-facing related work section structure for the hypercorpus paper.
Organized by research theme, not alphabetically. Each theme lists key papers, their relevance, and how hypercorpus differentiates.

References below use short names that map to `docs/literature-map.tsv` entries.

---

## 1. Multi-Hop Question Answering Benchmarks

This subsection establishes the evaluation landscape and motivates the problem. It should be brief (1-2 paragraphs in the final paper) since the datasets are introduced more fully in the experimental setup.

### Key papers

- **HotpotQA** (Yang et al., 2018, EMNLP). Multi-hop QA with supporting-fact annotations over Wikipedia. Defines the distractor and fullwiki settings that anchor most multi-hop retrieval work. The supporting-fact annotations provide the ground truth for hypercorpus's selector-first evaluation.
- **2WikiMultiHopQA** (Ho et al., 2020, COLING). Explicit reasoning-path annotations reduce pseudo-multi-hop shortcuts. Used as hypercorpus's calibration dataset.
- **MuSiQue** (Min et al., 2022, TACL). Harder multi-hop benchmark via single-hop composition with strict filtering against shortcut solvability. Tests whether hypercorpus's link-following genuinely covers required hops.
- **IIRC** (Ferguson et al., 2020, EMNLP). Reading comprehension where the initial paragraph is incomplete and the reader must follow links to external documents. The most naturally aligned benchmark for hypercorpus's "follow natural links to fill evidence gaps" hypothesis. Primary paper-facing dataset.
- **Compositional Questions Do Not Necessitate Multi-hop Reasoning** (Min et al., 2019, ACL). Critical paper showing many multi-hop questions are solvable via single-hop shortcuts. Essential context for why hypercorpus evaluates evidence coverage (support F1) rather than only answer correctness.
- **FRAMES** (Krishna et al., 2024, arXiv). Multi-hop RAG benchmark that tests factuality, retrieval, and reasoning jointly. SOTA models achieve 0.40 accuracy without retrieval, 0.66 with multi-step retrieval. Does not include explicit budget/efficiency metrics, but supports the problem motivation that multi-hop retrieval quality remains a bottleneck.

### Differentiation from hypercorpus

These benchmarks define the evaluation surface but not the method. Webwalker treats them as selector-evaluation contexts: the goal is compact evidence recovery under budget, not answer generation. The paper should note that IIRC's natural hyperlink structure makes it the most ecologically valid benchmark for the approach.

### Citation priority

The first five are mandatory citations. HotpotQA, IIRC, and MuSiQue appear in the experiments; 2WikiMultiHopQA appears as calibration. Min et al. (2019) is needed to justify the evidence-first evaluation design. FRAMES is a supporting citation for the problem motivation.

---

## 2. Dense Retrieval for Multi-Hop QA

This is the primary baseline family. The paper must clearly establish what flat and iterative dense retrieval can do before claiming hyperlink-aware selection adds value.

### Key papers

- **MDR** (Xiong et al., 2021, ICLR). Multi-hop dense retrieval: iteratively encodes the query plus previously retrieved passages to fetch the next hop. The strongest retrieval-centric baseline that does not use explicit link structure. Mandatory direct comparator.
- **HopRetriever** (Li et al., 2020, arXiv). Explicitly models hop sequences in multi-hop retrieval, expanding the evidence set step by step. Relevant as a hop-aware dense retrieval variant.
- **Iterative Document Reranking** (Zhang et al., 2021, SIGIR). Unified framework for iterative retrieval, reranking, and adaptive stopping across any-hop questions. Provides a direct contrast for hypercorpus's budget-aware stopping strategy.
- **Multi-Hop Paragraph Retrieval** (Feldman and El-Yaniv, 2019, arXiv). Early iterative paragraph retrieval that feeds previous-hop evidence into next-hop query representations. Establishes the iterative retrieval paradigm.
- **Combining Lexical and Dense Retrieval** (Sidiropoulos et al., 2021, arXiv). Shows hybrid lexical+dense retrieval can approximate multi-hop dense retrieval at lower cost. Relevant reminder that hypercorpus's seed selection need not be purely dense.

### Differentiation from hypercorpus

These systems treat documents as isolated items in an embedding space and rely on learned or heuristic query reformulation to bridge hops. They do not exploit the link structure already present in the corpus. Webwalker's key difference is using natural hyperlinks and their local semantics (anchor text, sentence context) as the navigation substrate, rather than re-querying a flat index at each hop. Under the same token budget, this structural signal can recover bridge documents that flat dense retrieval misses.

### Citation priority

MDR is the single most important baseline citation. Iterative Document Reranking is important for the stopping-strategy contrast. HopRetriever strengthens the hop-aware retrieval comparison. The other two are supporting citations.

---

## 3. Graph-Based Retrieval and GraphRAG

This section positions hypercorpus relative to systems that also use graph structure, but differ in how the graph is constructed and queried.

### Key papers

- **GraphRetriever** (Asai et al., 2020, ICLR). Treats Wikipedia as a graph and trains a retriever to follow reasoning paths. The closest learned ancestor to hypercorpus in spirit. Key difference: GraphRetriever learns task-specific path policies, while hypercorpus uses zero-shot dense-started selection over existing hyperlink structure without per-dataset training.
- **HippoRAG** (Gutierrez et al., 2024, NeurIPS). Combines knowledge graphs, Personalized PageRank, and LLMs into a neurobiologically inspired long-term memory retrieval framework. Strong multi-hop QA performance. Key difference: HippoRAG builds an explicit knowledge graph eagerly, while hypercorpus operates directly on natural hyperlinks at query time without prior graph construction.
- **HippoRAG 2** (Gutierrez et al., 2025, arXiv 2502.14802). Extends HippoRAG with a multi-memory architecture and improved Personalized PageRank, framed as "non-parametric continual learning." Achieves 7% improvement over competing embedding models. Key difference: even with reduced construction overhead, HippoRAG 2 still requires indexing-time graph building, while hypercorpus operates on existing hyperlinks requiring no construction step. The "non-parametric continual learning" framing adds a new angle vs. hypercorpus's "budgeted corpus selection" framing.
- **EcphoryRAG** (Liao, 2025, arXiv 2510.08958). Entity-centric retrieval with 94% token reduction vs. comparable systems, improving average EM from 0.392 to 0.474 over HippoRAG. Directly competes with HippoRAG on the same benchmarks and claims token efficiency. Strengthens the narrative that token cost matters in graph-based RAG, which supports hypercorpus's budget-awareness emphasis.
- **GNN-RAG** (Mavromatis and Karypis, 2025, ACL). Uses GNNs to reason over knowledge graphs, then retrieves shortest paths as LLM context. Represents the learned graph-reasoning approach. Key difference: GNN-RAG requires trained GNN components, while hypercorpus uses zero-shot scoring over existing hyperlink structure. Strengthens the zero-shot vs. learned contrast in Section 3.
- **KG2RAG** (Zhu et al., 2025, NAACL). Uses knowledge graphs to expand initial semantic retrieval chunks and reorganize context. Shares hypercorpus's "semantic seed then graph expansion" skeleton but operates over constructed KGs rather than natural hyperlinks.
- **Walk&Retrieve** (Bockling et al., 2025, IR-RAG@SIGIR). Zero-shot retrieval via knowledge graph walks. The closest system to hypercorpus's walk-based approach, but its graph is an explicit KG rather than a natural hyperlink graph.
- **PathRAG** (Chen et al., 2025, arXiv). Prunes graph-based RAG with relational paths. Relevant for path-scoring and path-pruning design decisions.
- **GraphRAG survey** (Han et al., 2025, arXiv). Comprehensive survey of the GraphRAG design space (query processor, retriever, organizer, generator). Useful for positioning hypercorpus within the broader taxonomy.

### Secondary graph systems (boundary-defining, not direct comparators)

- **RAPTOR** (Sarthi et al., 2024, arXiv). Recursive abstractive tree-organized retrieval. Represents the "eager global structuring" approach that hypercorpus avoids.
- **LightRAG** (Guo et al., 2024, arXiv). Efficiency-oriented GraphRAG with graph-structured indexing. Useful as an engineering-efficiency reference point.
- **LEGO-GraphRAG** (Cao et al., 2025, PVLDB). Modular decomposition of GraphRAG workflow with systematic classification of techniques. Provides empirical tradeoff studies on reasoning quality vs. efficiency. Useful for positioning hypercorpus's approach within the broader GraphRAG design space at a top database venue.
- **Hierarchical Lexical Graph** (Ghassel et al., 2025, KDD). Lexical graph hierarchy for multi-hop retrieval. Graph-structured multi-hop retrieval at a top venue. Different from hypercorpus (lexical graph vs. natural hyperlink graph) but occupies the same problem space.
- **Graph-O1** (Liu, 2025, arXiv). MCTS + RL for stepwise graph reasoning with selective subgraph retrieval. High cost (RL training + MCTS inference), which strengthens hypercorpus's lightweight-walk argument by contrast.

### Differentiation from hypercorpus

Webwalker differs from all graph-based systems on three axes: (1) it does not require eager graph construction -- the hyperlink graph already exists in the corpus; (2) it operates at query time with zero-shot scoring rather than requiring per-dataset training; (3) it enforces explicit token budgets as a first-class constraint. The paper should frame hypercorpus as a lighter-weight alternative that exploits existing link structure rather than building new graph structure.

### Citation priority

GraphRetriever and HippoRAG/HippoRAG 2 are the two most important narrative comparators. GNN-RAG and EcphoryRAG strengthen the contrast between learned and zero-shot graph approaches. GraphRAG survey and KG2RAG are important for positioning. Walk&Retrieve is important because of its methodological proximity. LEGO-GraphRAG and Hierarchical Lexical Graph ensure coverage of current top-venue work. RAPTOR, LightRAG, and Graph-O1 are boundary-defining citations.

---

## 4. Budget-Aware and Adaptive Retrieval

This is the novel niche where hypercorpus makes its most distinctive contribution. Explicit token-budget control over evidence *discovery via graph traversal* is novel, though token-budget-aware post-retrieval selection is an emerging research area.

### 4A. Adaptive retrieval controllers

- **Adaptive-RAG** (Jeong et al., 2024, NAACL). Dynamically selects no-retrieval, single-shot, or iterative retrieval based on question complexity. Relevant as an adaptive controller, but adapts the retrieval strategy, not the token budget of the selected evidence.
- **Self-RAG** (Asai et al., 2024, ICLR). Learns when to retrieve, how to critique, and when to stop via self-reflection tokens. The strongest learned-controller comparator for hypercorpus's "when to stop walking" decision, but operates at the generation level, not at the pre-RAG selection level.
- **IRCoT** (Trivedi et al., 2022, arXiv). Interleaves chain-of-thought with retrieval, using reasoning output to guide the next retrieval step. Key iterative retrieval comparator. Differs from hypercorpus in that it uses CoT-generated queries rather than hyperlink structure to decide the next step.

### 4B. Token-budget-aware context selection (post-retrieval)

This is the most important sub-area for novelty differentiation. These systems operate on already-retrieved flat passage sets, selecting what to include under a token budget. Webwalker differs by performing budget-aware *graph discovery* -- deciding which links to follow and when to stop walking -- rather than post-retrieval compression.

- **AdaGReS** (Peng et al., 2025, arXiv 2512.25052). Greedy context selection under token budget with submodularity guarantees. Balances relevance and redundancy with theoretical near-optimality proof. **The most direct differentiator in Section 4.** Key distinction from hypercorpus: AdaGReS operates on already-retrieved flat passage sets (post-retrieval selection), while hypercorpus performs budget-aware graph discovery. The submodularity framing is worth engaging with in the paper.
- **Stronger Baselines for RAG with Long-Context LMs** (Laitenberger, Manning, and Liu, 2025, arXiv 2506.03989). Systematically evaluates RAG under scaled token budgets. Finds that simple document-order-preserving single-stage retrieval (DOS RAG) matches or beats complex pipelines (ReadAgent, RAPTOR). Their "just scale the budget" finding directly challenges complex selection schemes. Webwalker must argue that budget-aware graph traversal provides value beyond simply stuffing more tokens.
- **Fishing for Answers** (Lin et al., 2025, arXiv 2509.04820). Compares one-shot adaptive chunk selection under token budget vs. iterative agentic retrieval for legal/regulatory QA. Directly compares one-shot-under-budget vs. iterative strategies, which maps onto hypercorpus's budget-constrained walk vs. MDR-style iteration.
- **SF-RAG** (Yu et al., 2026, arXiv 2602.13647). Structure-fidelity retrieval for academic QA. Preserves hierarchical paper structure during retrieval with path-guided retrieval under fixed token budget. Analogous to hypercorpus exploiting document link structure, but for hierarchical (tree) rather than graph (hyperlink) structure.
- **Core-based Hierarchies for Efficient GraphRAG** (Hossain and Sariyuce, 2026, arXiv 2603.05207). Replaces Leiden clustering with k-core decomposition for deterministic GraphRAG hierarchies. Implements token-budget-aware sampling strategy. Different approach (community-based summarization) but shares the budget-awareness theme.

### 4C. Agentic reasoning-integrated retrieval

A rapidly growing paradigm where retrieval is embedded as an action within LLM reasoning chains. These systems compete with hypercorpus's structured-graph-walk approach but spend token budget on reasoning tokens rather than evidence tokens.

- **Search-o1** (Li et al., 2025, EMNLP). Integrates agentic RAG into reasoning chains with a document analysis module for noise filtering. Represents the reasoning+retrieval fusion paradigm. The comparison point: hypercorpus uses corpus structure (hyperlinks) as navigation signal, while Search-o1 uses LLM reasoning chains to decide what to retrieve next from a flat index.
- **R1-Searcher** (Song et al., 2025, arXiv 2503.05592). Uses reinforcement learning to train LLM search capability, enabling autonomous decisions about when and what to retrieve during reasoning. Part of the agentic retrieval family alongside Search-o1 and ReSearch.
- **ReSearch** (Chen et al., 2025, arXiv 2503.19470). RL-trained reasoning+search integration. Completes the Search-o1 / R1-Searcher / ReSearch family of agentic retrieval systems.
- **MemoRAG** (Qian et al., 2024, arXiv 2409.05591). Memory-inspired knowledge discovery with a global memory module for retrieval clue generation followed by targeted retrieval. Structurally parallel to hypercorpus's dense-seed-then-walk approach, but operates over flat corpora rather than linked graphs.
- **OneGen** (Zhang et al., 2024, EMNLP Findings). Unifies generation and retrieval in a single forward pass. Represents the opposite architectural choice from hypercorpus's selector-first design. Useful as a "why not end-to-end?" foil in the related work.

### Differentiation from hypercorpus

The key novelty claim should be refined: **budget-aware evidence *discovery via graph traversal* is novel**. The distinction has two levels:

1. **vs. post-retrieval budget selectors (AdaGReS, Stronger Baselines, SF-RAG)**: These systems optimize what to include from an already-retrieved passage set. Webwalker's budget constraint operates at the *discovery/selection* level -- deciding which links to follow and when to stop walking -- not at the context compression level. The `budget_fill_relative_drop` stopping criterion operates during walk expansion, not during context packing.

2. **vs. agentic retrieval systems (Search-o1, R1-Searcher, ReSearch)**: These systems use LLM calls at every retrieval step, spending budget on reasoning tokens. Webwalker's lightweight scoring (anchor overlap, embedding similarity) keeps costs low. Under a fixed token budget, agentic systems spend budget on reasoning tokens, while hypercorpus spends budget on evidence tokens.

### Citation priority

AdaGReS is the single most important new citation for this section -- it is the most direct threat to the novelty claim and must be carefully differentiated. IRCoT remains the most direct iterative comparator. Stronger Baselines is important because it challenges the value of complex selection under scaled budgets. Adaptive-RAG and Self-RAG frame the adaptive-retrieval design space. Search-o1 represents the agentic paradigm. The other agentic papers (R1-Searcher, ReSearch) and budget papers (Fishing for Answers, SF-RAG, Core-based GraphRAG) are supporting citations.

---

## 5. Corpus and Subgraph Selection

This section covers prior work on selecting subsets of corpora or subgraphs before downstream processing, which is the most direct framing of hypercorpus's problem.

### Key papers

- **HGN** (Fang et al., 2020, EMNLP). Hierarchical Graph Network builds multi-granularity graphs (question, paragraph, sentence, entity) and uses graph propagation for joint support-fact and answer prediction. Relevant as a post-retrieval graph reasoning method that assumes the relevant documents are already retrieved.
- **Select, Answer and Explain** (Tu et al., 2020, AAAI). Explicit document selection before joint answer and supporting-sentence prediction. The most direct "select-then-read" predecessor, but without budget constraints or hyperlink-aware selection.
- **Two-stage Selector and Reader (FE2H)** (Li et al., 2022, ICASSP). Two-stage selector-reader without graph modules. Reminds the paper to justify why hyperlink-aware selection outperforms simpler two-stage filtering.
- **Exploiting Relevance Feedback in Knowledge Graph Search** (Su et al., 2015, KDD). Query-conditioned graph discovery with relevance feedback on knowledge graphs. Establishes that query-time graph discovery is a standalone algorithmic problem, not only a downstream QA helper.
- **Keyword Search over Knowledge Graphs via Static and Dynamic Hub Labelings** (Shi et al., 2020, WWW). Efficient compact-subgraph discovery on large knowledge graphs. Demonstrates that query-time subgraph discovery can be formulated as an independent algorithmic contribution.

### Differentiation from hypercorpus

Prior corpus/subgraph selection work either operates on constructed knowledge graphs (not natural hyperlinks), does not enforce explicit token budgets, or treats selection as a preprocessing step rather than the primary contribution. Webwalker unifies these threads: it performs query-time subgraph discovery over natural hyperlink structure under explicit token-budget constraints, with selection quality as the primary evaluation target.

### Citation priority

HGN and Select-Answer-Explain are important for the "select-then-read" lineage. Su et al. (2015) and Shi et al. (2020) are important for framing subgraph discovery as a standalone algorithmic problem, especially if targeting KDD or WWW.

---

## 6. Web Navigation and Focused Crawling

This section provides the adjacent-domain inspiration for hypercorpus's hyperlink-following approach.

### Key papers

- **End-to-End Goal-Driven Web Navigation** (Nogueira and Cho, 2016, NeurIPS). Models websites as page-and-hyperlink graphs, training an agent to navigate by following links toward a natural-language goal. The most direct adjacent ancestor for hypercorpus's "follow links toward a query" paradigm, but framed as RL navigation rather than evidence retrieval.
- **Focused Crawling** (Chakrabarti et al., 1999, Computer Networks). Classic work on topic-specific web crawling using classifiers to prioritize link expansion. Establishes the "selective link following" paradigm that hypercorpus inherits.
- **TREC Complex Answer Retrieval (CAR)** (Nanni et al., 2017, ICTIR). Benchmark for retrieving passages that compose into complex answers. The closest IR-community formulation to hypercorpus's "evidence discovery" problem, though without explicit hyperlink navigation.
- **Characterizing Question Facets for CAR** (MacAvaney et al., 2018, SIGIR). Facet-utility modeling for complex answer retrieval. Relevant for making hypercorpus's link-expansion decisions sensitive to query structure and facet utility rather than flat similarity.
- **Topic-Sensitive PageRank** (Haveliwala, 2002, WWW). Query-biased PageRank scores. Classic reference for query-conditioned graph priors, relevant to hypercorpus's dense-started seed selection.
- **HLP** (Zhou et al., 2022, arXiv 2203.06942). Hyperlink-induced pre-training for passage retrieval. Uses hyperlink topology for dense retriever pre-training. Validates that hyperlinks as retrieval signals is an established idea.
- **PHP** (Wu et al., 2022, arXiv 2209.06583). Progressive hyperlink prediction for IR pre-training. Together with HLP, shows that hyperlinks were explored for pre-training but not for budgeted selection -- which strengthens hypercorpus's distinctive contribution.

### Differentiation from hypercorpus

Web navigation and crawling work operates on open web graphs with broader objectives (topic harvesting, goal completion). Hyperlink pre-training work (HLP, PHP) uses link topology to improve dense retrievers at training time but does not use hyperlinks at query time for evidence discovery. Webwalker adapts the link-following paradigm to a retrieval-evaluation setting: the graph is a bounded corpus with natural hyperlinks, the objective is compact evidence recovery, and success is measured by support F1 under token budgets rather than by page harvest rate or task completion.

### Citation priority

Nogueira and Cho (2016) is the most important single citation for the navigation angle. Focused Crawling (1999) is important as a classic ancestor. TREC CAR and the facet-utility paper are important for the IR-native framing. Topic-Sensitive PageRank is a supporting citation.

---

## Writing Guidelines

### Narrative flow

The final related work section should follow this order: (1) briefly establish the multi-hop QA evaluation landscape, (2) cover dense and iterative retrieval as the primary baseline family, (3) discuss graph-based retrieval as the closest systems conceptually (now including HippoRAG 2, GNN-RAG, EcphoryRAG at top venues), (4) highlight the budget-aware retrieval landscape with clear differentiation between post-retrieval budget selection (AdaGReS), budget scaling (Stronger Baselines), agentic retrieval (Search-o1 family), and hypercorpus's discovery-time budget control, (5) note the corpus/subgraph selection lineage, and (6) briefly mention the web-navigation inspiration. Theme 4 is now richer and requires a more nuanced differentiation than before; the novelty argument is "budget-aware discovery via graph traversal" rather than "budget-aware retrieval is entirely new."

### What to emphasize

- Dense retrieval, iterative retrieval, graph retrieval, and LLM-guided retrieval are established design spaces. Write about them as known ingredients.
- Webwalker's distinctive combination is: dense-started evidence assembly + natural hyperlink locality + explicit token-budget control + selector-first evaluation. This combination is novel even though each ingredient has precedent.
- `budget_fill_relative_drop` is the cleanest candidate for a named new component.

### What not to overclaim

- Do not write as if the paper already beats all graph-based or learned retrievers (GraphRetriever, HippoRAG/HippoRAG 2 are not yet directly compared).
- Do not describe `mdr_light` as direct `MDR` -- the paper must use real MDR for the comparison.
- Do not let related work pull the paper into a full QA-system or end-to-end RAG comparison frame.
- Do not claim budget-aware retrieval is entirely unprecedented -- AdaGReS, Stronger Baselines, and others directly address token-budget-aware context assembly. The refined claim is that explicit **token-budget-constrained evidence *discovery via graph traversal*** is novel. Post-retrieval budget-aware selection (AdaGReS) and budget scaling studies (Stronger Baselines) are acknowledged as related but operate at a different level (post-retrieval compression vs. discovery-time traversal).
- Do not ignore the agentic retrieval paradigm (Search-o1, R1-Searcher, ReSearch) -- it must be acknowledged as a competing approach with a clear cost/budget differentiation.

### Approximate length

In the final SIGIR paper (8 pages, double-column), the related work section should be approximately 1-1.5 pages. Each theme above maps to 1-2 paragraphs.
