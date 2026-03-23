# Hypercorpus Related Work Outline

Purpose: paper-facing related work section structure for the webwalker paper.
Organized by research theme, not alphabetically. Each theme lists key papers, their relevance, and how webwalker differentiates.

References below use short names that map to `docs/literature-map.tsv` entries.

---

## 1. Multi-Hop Question Answering Benchmarks

This subsection establishes the evaluation landscape and motivates the problem. It should be brief (1-2 paragraphs in the final paper) since the datasets are introduced more fully in the experimental setup.

### Key papers

- **HotpotQA** (Yang et al., 2018, EMNLP). Multi-hop QA with supporting-fact annotations over Wikipedia. Defines the distractor and fullwiki settings that anchor most multi-hop retrieval work. The supporting-fact annotations provide the ground truth for webwalker's selector-first evaluation.
- **2WikiMultiHopQA** (Ho et al., 2020, COLING). Explicit reasoning-path annotations reduce pseudo-multi-hop shortcuts. Used as webwalker's calibration dataset.
- **MuSiQue** (Min et al., 2022, TACL). Harder multi-hop benchmark via single-hop composition with strict filtering against shortcut solvability. Tests whether webwalker's link-following genuinely covers required hops.
- **IIRC** (Ferguson et al., 2020, EMNLP). Reading comprehension where the initial paragraph is incomplete and the reader must follow links to external documents. The most naturally aligned benchmark for webwalker's "follow natural links to fill evidence gaps" hypothesis. Primary paper-facing dataset.
- **Compositional Questions Do Not Necessitate Multi-hop Reasoning** (Min et al., 2019, ACL). Critical paper showing many multi-hop questions are solvable via single-hop shortcuts. Essential context for why webwalker evaluates evidence coverage (support F1) rather than only answer correctness.

### Differentiation from webwalker

These benchmarks define the evaluation surface but not the method. Webwalker treats them as selector-evaluation contexts: the goal is compact evidence recovery under budget, not answer generation. The paper should note that IIRC's natural hyperlink structure makes it the most ecologically valid benchmark for the approach.

### Citation priority

All five are mandatory citations. HotpotQA, IIRC, and MuSiQue appear in the experiments; 2WikiMultiHopQA appears as calibration. Min et al. (2019) is needed to justify the evidence-first evaluation design.

---

## 2. Dense Retrieval for Multi-Hop QA

This is the primary baseline family. The paper must clearly establish what flat and iterative dense retrieval can do before claiming hyperlink-aware selection adds value.

### Key papers

- **MDR** (Xiong et al., 2021, ICLR). Multi-hop dense retrieval: iteratively encodes the query plus previously retrieved passages to fetch the next hop. The strongest retrieval-centric baseline that does not use explicit link structure. Mandatory direct comparator.
- **HopRetriever** (Li et al., 2020, arXiv). Explicitly models hop sequences in multi-hop retrieval, expanding the evidence set step by step. Relevant as a hop-aware dense retrieval variant.
- **Iterative Document Reranking** (Zhang et al., 2021, SIGIR). Unified framework for iterative retrieval, reranking, and adaptive stopping across any-hop questions. Provides a direct contrast for webwalker's budget-aware stopping strategy.
- **Multi-Hop Paragraph Retrieval** (Feldman and El-Yaniv, 2019, arXiv). Early iterative paragraph retrieval that feeds previous-hop evidence into next-hop query representations. Establishes the iterative retrieval paradigm.
- **Combining Lexical and Dense Retrieval** (Sidiropoulos et al., 2021, arXiv). Shows hybrid lexical+dense retrieval can approximate multi-hop dense retrieval at lower cost. Relevant reminder that webwalker's seed selection need not be purely dense.

### Differentiation from webwalker

These systems treat documents as isolated items in an embedding space and rely on learned or heuristic query reformulation to bridge hops. They do not exploit the link structure already present in the corpus. Webwalker's key difference is using natural hyperlinks and their local semantics (anchor text, sentence context) as the navigation substrate, rather than re-querying a flat index at each hop. Under the same token budget, this structural signal can recover bridge documents that flat dense retrieval misses.

### Citation priority

MDR is the single most important baseline citation. Iterative Document Reranking is important for the stopping-strategy contrast. HopRetriever strengthens the hop-aware retrieval comparison. The other two are supporting citations.

---

## 3. Graph-Based Retrieval and GraphRAG

This section positions webwalker relative to systems that also use graph structure, but differ in how the graph is constructed and queried.

### Key papers

- **GraphRetriever** (Asai et al., 2020, ICLR). Treats Wikipedia as a graph and trains a retriever to follow reasoning paths. The closest learned ancestor to webwalker in spirit. Key difference: GraphRetriever learns task-specific path policies, while webwalker uses zero-shot dense-started selection over existing hyperlink structure without per-dataset training.
- **HippoRAG** (Gutierrez et al., 2024, NeurIPS). Combines knowledge graphs, Personalized PageRank, and LLMs into a neurobiologically inspired long-term memory retrieval framework. Strong multi-hop QA performance. Key difference: HippoRAG builds an explicit knowledge graph eagerly, while webwalker operates directly on natural hyperlinks at query time without prior graph construction.
- **KG2RAG** (Zhu et al., 2025, NAACL). Uses knowledge graphs to expand initial semantic retrieval chunks and reorganize context. Shares webwalker's "semantic seed then graph expansion" skeleton but operates over constructed KGs rather than natural hyperlinks.
- **Walk&Retrieve** (Bockling et al., 2025, IR-RAG@SIGIR). Zero-shot retrieval via knowledge graph walks. The closest system to webwalker's walk-based approach, but its graph is an explicit KG rather than a natural hyperlink graph.
- **PathRAG** (Chen et al., 2025, arXiv). Prunes graph-based RAG with relational paths. Relevant for path-scoring and path-pruning design decisions.
- **GraphRAG survey** (Han et al., 2025, arXiv). Comprehensive survey of the GraphRAG design space (query processor, retriever, organizer, generator). Useful for positioning webwalker within the broader taxonomy.

### Secondary graph systems (boundary-defining, not direct comparators)

- **RAPTOR** (Sarthi et al., 2024, arXiv). Recursive abstractive tree-organized retrieval. Represents the "eager global structuring" approach that webwalker avoids.
- **LightRAG** (Guo et al., 2024, arXiv). Efficiency-oriented GraphRAG with graph-structured indexing. Useful as an engineering-efficiency reference point.

### Differentiation from webwalker

Webwalker differs from all graph-based systems on three axes: (1) it does not require eager graph construction -- the hyperlink graph already exists in the corpus; (2) it operates at query time with zero-shot scoring rather than requiring per-dataset training; (3) it enforces explicit token budgets as a first-class constraint. The paper should frame webwalker as a lighter-weight alternative that exploits existing link structure rather than building new graph structure.

### Citation priority

GraphRetriever and HippoRAG are the two most important narrative comparators. GraphRAG survey and KG2RAG are important for positioning. Walk&Retrieve is important because of its methodological proximity. RAPTOR and LightRAG are boundary-defining citations.

---

## 4. Budget-Aware and Adaptive Retrieval

This is the novel niche where webwalker makes its most distinctive contribution. Explicit token-budget control over evidence assembly is underexplored in the literature.

### Key papers

- **Adaptive-RAG** (Jeong et al., 2024, NAACL). Dynamically selects no-retrieval, single-shot, or iterative retrieval based on question complexity. Relevant as an adaptive controller, but adapts the retrieval strategy, not the token budget of the selected evidence.
- **Self-RAG** (Asai et al., 2024, ICLR). Learns when to retrieve, how to critique, and when to stop via self-reflection tokens. The strongest learned-controller comparator for webwalker's "when to stop walking" decision, but operates at the generation level, not at the pre-RAG selection level.
- **IRCoT** (Trivedi et al., 2022, arXiv). Interleaves chain-of-thought with retrieval, using reasoning output to guide the next retrieval step. Key iterative retrieval comparator. Differs from webwalker in that it uses CoT-generated queries rather than hyperlink structure to decide the next step.

### Differentiation from webwalker

No existing system treats explicit token-budget-constrained evidence selection as the primary problem. Adaptive-RAG and Self-RAG adapt retrieval behavior but do not impose or optimize for a fixed token budget on the selected evidence set. IRCoT interleaves retrieval and reasoning but does not account for cumulative token cost. Webwalker's `budget_fill_relative_drop` and its budget-aware stopping are the cleanest novel components in this space.

### Citation priority

All three are important. IRCoT is the most direct iterative comparator. Adaptive-RAG and Self-RAG frame the adaptive-retrieval design space that webwalker extends with explicit budget control.

---

## 5. Corpus and Subgraph Selection

This section covers prior work on selecting subsets of corpora or subgraphs before downstream processing, which is the most direct framing of webwalker's problem.

### Key papers

- **HGN** (Fang et al., 2020, EMNLP). Hierarchical Graph Network builds multi-granularity graphs (question, paragraph, sentence, entity) and uses graph propagation for joint support-fact and answer prediction. Relevant as a post-retrieval graph reasoning method that assumes the relevant documents are already retrieved.
- **Select, Answer and Explain** (Tu et al., 2020, AAAI). Explicit document selection before joint answer and supporting-sentence prediction. The most direct "select-then-read" predecessor, but without budget constraints or hyperlink-aware selection.
- **Two-stage Selector and Reader (FE2H)** (Li et al., 2022, ICASSP). Two-stage selector-reader without graph modules. Reminds the paper to justify why hyperlink-aware selection outperforms simpler two-stage filtering.
- **Exploiting Relevance Feedback in Knowledge Graph Search** (Su et al., 2015, KDD). Query-conditioned graph discovery with relevance feedback on knowledge graphs. Establishes that query-time graph discovery is a standalone algorithmic problem, not only a downstream QA helper.
- **Keyword Search over Knowledge Graphs via Static and Dynamic Hub Labelings** (Shi et al., 2020, WWW). Efficient compact-subgraph discovery on large knowledge graphs. Demonstrates that query-time subgraph discovery can be formulated as an independent algorithmic contribution.

### Differentiation from webwalker

Prior corpus/subgraph selection work either operates on constructed knowledge graphs (not natural hyperlinks), does not enforce explicit token budgets, or treats selection as a preprocessing step rather than the primary contribution. Webwalker unifies these threads: it performs query-time subgraph discovery over natural hyperlink structure under explicit token-budget constraints, with selection quality as the primary evaluation target.

### Citation priority

HGN and Select-Answer-Explain are important for the "select-then-read" lineage. Su et al. (2015) and Shi et al. (2020) are important for framing subgraph discovery as a standalone algorithmic problem, especially if targeting KDD or WWW.

---

## 6. Web Navigation and Focused Crawling

This section provides the adjacent-domain inspiration for webwalker's hyperlink-following approach.

### Key papers

- **End-to-End Goal-Driven Web Navigation** (Nogueira and Cho, 2016, NeurIPS). Models websites as page-and-hyperlink graphs, training an agent to navigate by following links toward a natural-language goal. The most direct adjacent ancestor for webwalker's "follow links toward a query" paradigm, but framed as RL navigation rather than evidence retrieval.
- **Focused Crawling** (Chakrabarti et al., 1999, Computer Networks). Classic work on topic-specific web crawling using classifiers to prioritize link expansion. Establishes the "selective link following" paradigm that webwalker inherits.
- **TREC Complex Answer Retrieval (CAR)** (Nanni et al., 2017, ICTIR). Benchmark for retrieving passages that compose into complex answers. The closest IR-community formulation to webwalker's "evidence discovery" problem, though without explicit hyperlink navigation.
- **Characterizing Question Facets for CAR** (MacAvaney et al., 2018, SIGIR). Facet-utility modeling for complex answer retrieval. Relevant for making webwalker's link-expansion decisions sensitive to query structure and facet utility rather than flat similarity.
- **Topic-Sensitive PageRank** (Haveliwala, 2002, WWW). Query-biased PageRank scores. Classic reference for query-conditioned graph priors, relevant to webwalker's dense-started seed selection.

### Differentiation from webwalker

Web navigation and crawling work operates on open web graphs with broader objectives (topic harvesting, goal completion). Webwalker adapts the link-following paradigm to a retrieval-evaluation setting: the graph is a bounded corpus with natural hyperlinks, the objective is compact evidence recovery, and success is measured by support F1 under token budgets rather than by page harvest rate or task completion.

### Citation priority

Nogueira and Cho (2016) is the most important single citation for the navigation angle. Focused Crawling (1999) is important as a classic ancestor. TREC CAR and the facet-utility paper are important for the IR-native framing. Topic-Sensitive PageRank is a supporting citation.

---

## Writing Guidelines

### Narrative flow

The final related work section should follow this order: (1) briefly establish the multi-hop QA evaluation landscape, (2) cover dense and iterative retrieval as the primary baseline family, (3) discuss graph-based retrieval as the closest systems conceptually, (4) highlight the budget-aware retrieval gap, (5) note the corpus/subgraph selection lineage, and (6) briefly mention the web-navigation inspiration. Themes 4 and 5 are where webwalker's novelty is strongest.

### What to emphasize

- Dense retrieval, iterative retrieval, graph retrieval, and LLM-guided retrieval are established design spaces. Write about them as known ingredients.
- Webwalker's distinctive combination is: dense-started evidence assembly + natural hyperlink locality + explicit token-budget control + selector-first evaluation. This combination is novel even though each ingredient has precedent.
- `budget_fill_relative_drop` is the cleanest candidate for a named new component.

### What not to overclaim

- Do not write as if the paper already beats all graph-based or learned retrievers (GraphRetriever, HippoRAG are not yet directly compared).
- Do not describe `mdr_light` as direct `MDR` -- the paper must use real MDR for the comparison.
- Do not let related work pull the paper into a full QA-system or end-to-end RAG comparison frame.
- Do not claim budget-aware retrieval is entirely unprecedented -- Adaptive-RAG and Self-RAG touch on adaptive behavior. The claim is that explicit token-budget-constrained evidence selection as the primary problem formulation is novel.

### Approximate length

In the final SIGIR paper (8 pages, double-column), the related work section should be approximately 1-1.5 pages. Each theme above maps to 1-2 paragraphs.
