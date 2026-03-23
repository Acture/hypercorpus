## 2. Problem Formulation

We formalize the task of **budgeted corpus selection** over naturally linked document collections. The goal is to assemble a compact evidence set from a hyperlinked corpus that maximizes support coverage for a given query, subject to an explicit token budget.

### 2.1 Linked Document Graph

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ denote a *link-context graph* over a document corpus. Each node $v \in \mathcal{V}$ represents a document with title $t_v$ and an ordered sentence sequence $\mathbf{s}_v = (s_v^1, \ldots, s_v^{n_v})$. The full text of a document is the concatenation of its sentences, and its token cost is $\tau(v) = |\text{tok}(\mathbf{s}_v)|$, the number of tokens under a fixed tokenizer.

Each directed edge $e = (u, v) \in \mathcal{E}$ corresponds to a hyperlink from document $u$ to document $v$. Crucially, every edge carries *link context*: a triple $\ell_e = (a_e, c_e, i_e)$, where $a_e$ is the anchor text of the hyperlink, $c_e$ is the sentence in $u$ containing the anchor, and $i_e$ is the sentence index within $u$. This link context is a distinguishing feature of naturally linked corpora such as Wikipedia: it provides local semantic signal about the relevance of a neighboring document *before* that document is retrieved. Multiple hyperlinks may exist between the same pair of documents, each with distinct context.

### 2.2 Budgeted Corpus Selection

Given a query $q$ and a link-context graph $\mathcal{G}$, the **budgeted corpus selection** problem seeks a subset $S^* \subseteq \mathcal{V}$ that maximizes evidence quality under a token budget $B$:

$$S^* = \arg\max_{S \subseteq \mathcal{V},\; \sum_{v \in S} \tau(v) \leq B} \text{F1}_\emptyset(S, G),$$

where $G$ is the set of gold support documents for $q$. A **corpus selector** $\sigma$ is any algorithm that produces a feasible solution $\sigma(q, \mathcal{G}) = S$ satisfying the budget constraint. The budget $B$ is specified in tokens and represents a hard upper bound on the total volume of text passed to a downstream consumer (e.g., a reader model or GraphRAG pipeline). The selector may additionally be constrained by a maximum number of expansion steps $H$ (hop budget) and a seed size $k$, collectively forming a **selector budget** $(B, H, k)$.

Selection proceeds in two stages. First, a **dense seed retrieval** step identifies up to $k$ initial documents from $\mathcal{V}$ by embedding similarity to $q$. Second, a **link-context expansion** step traverses edges in $\mathcal{G}$ starting from the seed set, using link-context scoring to decide which neighbors to include. The selector consumes at most $H$ expansion steps and must stay within the token budget $B$ at all times. After the walk terminates, a **budget fill** step may use the remaining token headroom $B - \sum_{v \in S} \tau(v)$ to include additional documents ranked by embedding similarity, ensuring high budget utilization.

### 2.3 Evaluation Criteria

We evaluate selection quality against a set of gold support documents $G \subseteq \mathcal{V}$ annotated per query. The primary metric is **support F1**, the harmonic mean of support precision and support recall:

$$\text{Precision}(S, G) = \frac{|S \cap G|}{|S|}, \quad \text{Recall}(S, G) = \frac{|S \cap G|}{|G|},$$

$$\text{F1}(S, G) = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}.$$

We adopt a strict variant, $\text{F1}_\emptyset$, that assigns a score of zero when the selected set is empty ($S = \emptyset$), penalizing selectors that fail to return any evidence. This is the headline metric throughout our experiments.

Two secondary metrics characterize budget behavior. **Budget adherence** is the binary indicator $\mathbb{1}[\sum_{v \in S} \tau(v) \leq B]$, verifying that the selector respects the hard constraint. **Budget utilization** is the ratio $\sum_{v \in S} \tau(v) / B$, measuring how effectively the selector fills the available budget. High utilization under high F1 indicates efficient use of the token allocation; low utilization suggests the selector terminates prematurely. We report support precision, support recall, and the number of selected nodes as additional diagnostics.

This formulation is deliberately **selector-first**: the objective is evidence assembly quality, not downstream answer correctness. Answer-level metrics (exact match, answer F1) are reported as secondary sanity checks but do not drive method design or comparison.
