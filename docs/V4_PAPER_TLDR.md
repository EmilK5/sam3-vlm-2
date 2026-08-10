# Closed-Loop Semantic Active Perception for Zero-Shot Counting

*Compact V4 research formulation / working paper draft. Intended as the human-readable scientific source of truth. The implementation specification lives separately in `V4_DESIGN_SPEC.md`.*

## Abstract

We study zero-shot counting in visually difficult scenes using a vision-language model (VLM) and a prompt-conditioned segmentation model. The main target is green-citrus counting, where green fruit and foliage are highly confusable, but the formulation is dataset-independent and also applies to cars, blood cells, and general counting benchmarks. Our central idea is to treat prompting not as one-shot detection but as **active semantic sensing**. A user first supplies the target concept. SAM3 executes this prompt globally to create a sensor-grounded candidate graph. Qwen then observes the original image together with representative SAM3 crops, confidence scores, and retrieval provenance, and proposes a bank of semantic sensing actions: target-oriented discovery prompts, confounder prompts, context prompts, and later verification prompts. The controller executes selected prompts globally, so one SAM3 pass can discover new objects and update beliefs for many existing objects at once. Qwen is called again only when the scene state changes meaningfully. Expensive per-object verification is reserved for the small residual set.

Mathematically, the system is a controlled sensing process over an evolving scene representation. Bajcsy's active-perception view supplies the outer principle: choose what, where, and how to sense in order to reduce task loss under sensing cost. Active Sequential Hypothesis Testing (ASHT) supplies the inner machinery for belief updates, adaptive experiment selection, and sequential stopping. Because counting additionally requires discovering objects not yet represented in the graph, we extend pure candidate-level ASHT with an explicit discovery term. Qwen is not the Bayesian controller; it amortizes the otherwise intractable search over natural-language sensing actions, while empirical SAM3 observations increasingly override its semantic prior.

## 1. Problem

Given an image $I$ and a user concept $c^\star$, the goal is to estimate the number of physical instances belonging to the target class $k^\star$, without task-specific training labels. Let the latent physical scene be

$$
\mathcal O^\star=\{o_1^\star,\ldots,o_{N^\star}^\star\},
$$

where each object has class $z_j$, geometry $g_j$, and latent visual attributes $a_j$. The true count is

$$
N^\star_{k^\star}=\sum_j \mathbf 1[z_j=k^\star].
$$

The main difficulty is that the object set itself is unknown. A system can be highly confident about every object it has discovered and still undercount badly because some target objects never entered the graph. Therefore counting requires two coupled inference problems:

$$
\boxed{\text{discover the relevant objects} + \text{resolve what they are}.}
$$

This matters strongly for green citrus: a one-shot prompt can miss occluded fruit and can also retrieve leaves whose color and local appearance resemble fruit.

## 2. Evolving scene belief

At sensing step $t$, the system maintains a working graph $G_t$ of sensor-grounded object hypotheses. It is not equated with the true scene. Graph nodes can be false positives, duplicates, or merges, while real objects can still be absent.

For each represented node $i$, the system maintains a semantic posterior

$$
\rho_{i,t}(k)=P(Z_i=k\mid H_t), \qquad \rho_{i,t}\in\Delta^{K-1},
$$

where $H_t$ is the sensing history. For green citrus, a useful class set is

$$
\mathcal K=\{\text{citrus},\text{leaf},\text{background}\}.
$$

The system also needs a representation of residual discovery uncertainty. Let

$$
M_t^\star
$$

be the latent number of target objects not yet represented in $G_t$. Conceptually,

$$
N^\star_{k^\star}=N^\star_{t,\mathrm{represented}}+M_t^\star.
$$

An exact posterior over $M_t^\star$ is difficult to obtain zero-shot, so V4 initially maintains an empirical discovery-potential state based on recent marginal discoveries, spatial coverage, tiled-versus-global gain, unresolved regions, and Qwen's assessment of missing appearance modes.

A compact operational state is

$$
B_t=(G_t,\rho_t,U_t,\mathcal S_t,\mathcal A_t,C_t),
$$

where $U_t$ denotes residual discovery potential, $\mathcal S_t$ semantic sensing history, $\mathcal A_t$ the current action bank, and $C_t$ consumed compute.

## 3. Semantic sensing actions

A sensing action is

$$
x=(\pi,\kappa,R,E^+,E^-,\tau,T),
$$

where $\pi$ is the text prompt, $\kappa$ a canonical semantic key, $R$ the region, $E^+,E^-$ positive/negative exemplars, $\tau$ the SAM3 threshold, and $T$ the spatial execution mode such as global or tiled inference.

The semantic key distinguishes the underlying semantic question from its exact wording. For example, "round green citrus" and "green spherical fruit" may be paraphrases of approximately the same semantic coordinate. Repeating such prompts can improve robustness, but should not be treated as independent semantic evidence.

To model class-semantic structure, retain the subset-moment view from the earlier formulation. Let $a_j\in\{0,1\}^V$ be latent visual attributes and, for a descriptor set $S\subseteq V$,

$$
\beta_{k,S}=P\!\left(\bigwedge_{w\in S} a_w=1\mid Z=k\right).
$$

The separation order between classes is

$$
r(k,l)=\min\{|S|:\beta_{k,S}\neq\beta_{l,S}\}.
$$

This captures the citrus/leaf regime: both may be green, so first-order attributes can be weak, while conjunctions such as "green and spherical" or "flat and veined" can separate them much more strongly.

SAM3 is the noisy sensor. For action $x$, its object-level response depends on the queried property through an action-specific channel $g_x$. In simplified form,

$$
P(Y_{i,x}=y\mid Z_i=k,x)
=
\beta_{k,\kappa(x)}g_x(y\mid1)
+
(1-\beta_{k,\kappa(x)})g_x(y\mid0).
$$

Qwen may initialize semantic usefulness $\beta$, but it does not determine the sensor channel $g_x$. Actual SAM3 behavior must be learned or calibrated from visual observations.

Crucially, SAM3 returns a **set** of detections for one global action,

$$
Y_x=\{(\hat g_r,s_r)\}_{r=1}^{m_x},
$$

not one scalar answer for one crop. After association to the graph, one global prompt can update many node beliefs and can create new nodes.

## 4. Bootstrap before Qwen

The first model action is always SAM3 using the user prompt directly:

$$
x_0=(\pi=c^\star,R=\Omega_I,T=\mathrm{global}).
$$

This stage is recall-oriented. The resulting detections are **candidates**, not positive labels. SAM3 confidence is retained as sensor evidence but is not equated with posterior target probability.

If object scale, image resolution, density, or weak coverage suggests that a global pass may miss small objects, the same semantic prompt may also be executed tiled. Global and tiled retrievals share the same semantic key, so agreement strengthens retrieval/localization support but is not counted as two independent semantic confirmations.

The bootstrap produces the initial graph and a bounded evidence pack for Qwen consisting of the original image plus a representative contact sheet. The sheet intentionally includes high-, medium-, and low-confidence candidates and appearance/spatial outliers. Each crop is annotated with SAM3 confidence and provenance such as "global only", "tiled only", or "global+tiled". Qwen is explicitly told that these are unverified sensor candidates. This avoids pseudoexemplar poisoning, where SAM3 false positives would otherwise be presented to Qwen as ground-truth positives.

## 5. Qwen as a scene-level semantic planner

Qwen is called **after** bootstrap because its job is not merely to list generic synonyms. It should reason about the interaction between the requested concept, the image, and SAM3's observed failure modes.

Given a compact planning representation $Q_t=\Phi(I,B_t,c^\star)$, Qwen proposes a finite action bank

$$
\mathcal A_t=G_\phi(Q_t,c^\star)\subset\mathcal X.
$$

The bank contains four families:

1. **Discovery actions:** alternative descriptions intended mainly to recover missed target appearances, e.g. "partially hidden green fruit".
2. **Confounder actions:** prompts for competing explanations, e.g. "flat green leaf" or "leaf cluster".
3. **Context actions:** prompts that localize useful search regions, e.g. "tree canopy".
4. **Verification actions:** fine-grained discriminative prompts, used mainly late in the run.

Each proposal contains a semantic key, prompt, family, suggested threshold/tiling, optional exemplar requirements, Qwen priority, and approximate class-semantic probabilities. Qwen's ranking is a prior, not the final decision rule.

Qwen should normally be called only 2--4 times per difficult image: after bootstrap, and again when the prompt bank is exhausted, marginal discovery stalls, a new confounder pattern emerges, or unresolved uncertainty remains substantial. This replaces the previous per-node Qwen loop.

## 6. Global vectorized multipass sensing

The fundamental computational unit is a **global semantic experiment**. Suppose Qwen proposes "green spherical fruit". SAM3 runs this prompt over the scene once. The resulting detections are associated with every relevant graph node, while unmatched detections may create new nodes. Therefore one SAM3 execution can simultaneously increase recall and change beliefs for dozens of candidates.

This is the central vectorization principle. If a global query gives only 0.1 bit of useful information to each of 50 nodes, its aggregate value can still be roughly 5 bits for one sensor execution. A crop-level query giving 0.7 bit to one object may be less efficient despite being individually stronger.

The approximate discriminative information of action $x$ is

$$
I_t(x)\approx\sum_{i\in V_t}w_{i,t}I(Z_i;O_{i,x}\mid B_t),
$$

where $O_{i,x}$ is the projected node-level observation and $w_{i,t}$ can emphasize uncertain nodes or nodes contributing strongly to count variance. The exact scene-level quantity is $I(Z_{V_t};\mathbf O_x\mid B_t)$; the additive form is a tractable approximation.

This changes the computational scaling from approximately "objects times verification rounds" to "number of useful global semantic experiments", which can remain small even in dense scenes.

## 7. Positive, confounder, and absence evidence

Positive and negative prompts are mathematically the same type of experiment. Their usefulness comes from different class-conditional likelihoods.

For citrus, a strong match to "green spherical fruit" favors citrus if

$$
\beta_{\mathrm{citrus},\kappa}\gg\beta_{\mathrm{leaf},\kappa}.
$$

A strong match to "flat green leaf" favors leaf if the reverse inequality holds. The latter is not an ad hoc negative veto; it is ordinary Bayesian evidence for a competing explanation.

Presence and absence should be asymmetric. SAM3 has imperfect recall, so failure to retrieve a node under a positive prompt is weaker evidence against the target than strong retrieval under a well-chosen confounder prompt is evidence for the alternative. Node observations should therefore remain soft (e.g. score or strong/weak/not-retrieved/not-observable), rather than using a symmetric $+1/-1$ rule.

This is especially important in the main green-citrus setting, where positive and refuting evidence naturally live in different semantic directions.

## 8. Active perception and ASHT

The overall architecture follows Bajcsy's active-perception principle: perception is controlled measurement. The image is fixed, but the system chooses what concept to ask for, where to search, how to tile, which exemplars to use, and when to stop. The feedback loop is

$$
B_t\rightarrow x_t\rightarrow Y_t\rightarrow B_{t+1}.
$$

ASHT supplies a more specific optimization view. In classical active hypothesis testing, the controller jointly selects experiments, updates posterior beliefs, and chooses a stopping time to trade sensing cost against decision error. For our represented objects, $\rho_{i,t}$ is the ASHT-like information state.

However, pure fixed-hypothesis ASHT is not enough because the object set is incomplete early in the run. We therefore combine three quantities:

$$
U_t(x)
=
\alpha_tD_t(x)
+
\beta_tI_t(x)
-
\gamma R_t(x)
-
\lambda C(x)
+
\eta_tQ_t(x).
$$

Here $D_t(x)$ is expected discovery gain, $I_t(x)$ discriminative or count-relevant information, $R_t(x)$ semantic redundancy, $C(x)$ computational cost, and $Q_t(x)$ Qwen's semantic priority. Qwen proposes a tractable candidate set; the controller chooses

$$
x_t^*=\arg\max_{x\in\mathcal A_t}U_t(x).
$$

Early in the run, discovery receives higher weight; later, discrimination dominates. Qwen's influence can decrease as empirical prompt performance becomes available.

ASHT contributes more than information gain. It supplies: (i) posterior beliefs as compact decision state; (ii) experiment selection based on class-dependent observation laws; (iii) sequential rather than fixed-budget stopping; and (iv) information-rate interpretations showing why better semantic experiments can reduce the number of required measurements. It also explains why confounder prompts matter: adaptive testing can gain when the best experiment for confirming one hypothesis differs from the best experiment for refuting it.

## 9. Qwen replanning

After several global passes, the scene presented to Qwen can be materially different from the bootstrap state. Newly discovered objects, repeated false-positive structures, and unresolved appearance clusters are summarized in a new contact sheet and compact prompt-performance table.

Qwen is called again only when useful: when the best remaining bank action has low utility, consecutive discovery gains plateau, a new confounder emerges, or substantial count uncertainty remains. The new action bank may explicitly target uncovered appearance modes rather than repeating the original semantic decomposition.

This yields closed-loop reasoning at the correct timescale:

$$
\text{Qwen strategy}\rightarrow\text{several sensor experiments}\rightarrow\text{changed scene}\rightarrow\text{Qwen replanning},
$$

rather than Qwen once per object.

## 10. Residual cleanup

Global sensing continues until most remaining uncertainty is concentrated in a small residual subset. Only then does V4 use expensive local verification.

The first cleanup action should still be vectorized where possible: assemble a contact sheet of ambiguous crops, let Qwen identify a common distinction, and execute the resulting prompt over the relevant ROIs as a batch. Individual object-query loops are the final fallback, not the normal operating mode.

This forms a computational funnel:

$$
\boxed{\text{global discovery}\rightarrow\text{global discrimination}\rightarrow\text{batched cleanup}\rightarrow\text{rare individual verification}.}
$$

## 11. Stopping and count

Stopping should not be "run $m$ prompts". In the ideal decision formulation, stop when the expected reduction in task loss from any available sensing action no longer exceeds its cost.

A generic Bellman view is

$$
V(B)=\min\left\{R_{\mathrm{stop}}(B),\;\min_x\left[C(x)+\mathbb E V(T(B,x,Y))\right]\right\}.
$$

V4 need not solve this POMDP exactly; its utility rule is an approximation to the same principle. Practical stopping requires both low residual discovery potential and sufficiently low count uncertainty, subject to hard Qwen/SAM3 budgets.

Under approximately correct registration, the represented soft count is

$$
\widehat N_{\mathrm{rep}}
=
\sum_{i\in V_T}e_i\rho_i(k^\star),
$$

where $e_i$ is node existence probability or an approximate registration weight. Conceptually, the full count also includes residual missed-object expectation:

$$
\mathbb E[N^\star_{k^\star}\mid H_T]
\approx
\widehat N_{\mathrm{rep}}+\mu_T^{\mathrm{miss}}.
$$

The first V4 implementation may report the represented soft count and a separate discovery-saturation diagnostic until $\mu_T^{\mathrm{miss}}$ can be calibrated reliably.

## 12. Main theoretical claims to investigate

The current formulation suggests several testable claims rather than assuming them as established facts.

**Global-query efficiency.** If one sensing action provides conditionally useful observations for many nodes while costing one SAM3 execution, scene-level information per unit compute can exceed crop-wise verification by a factor proportional to the number of affected uncertain nodes.

**Semantic separation.** If target and confounder classes match on low-order semantic moments but differ on higher-order conjunctions, single-attribute prompting can be uninformative while conjunction prompts provide positive information. This retains the separation-order argument from V3.

**Adaptive directional gain.** If the most discriminative target-confirming prompt differs from the most discriminative confounder-confirming prompt, feedback-dependent prompt choice can outperform a fixed one-direction prompt list, consistent with active hypothesis testing theory.

**Discovery diminishing returns.** Prompt-based discovery is a coverage process and should exhibit diminishing marginal discovery as semantic/spatial coverage saturates. This motivates greedy prompt selection and plateau-based replanning/stopping.

**Count-targeted sensing.** Because the terminal task is counting, an action can be valuable even if it does not fully classify any one node, provided that it reduces posterior uncertainty in the aggregate count.

These claims should be stated formally only after assumptions and observation models are fixed and then tested empirically through ablations.

## 13. Experiments and ablations

The core evaluation must separate **discovery**, **semantic resolution**, and **compute**. Relevant datasets include the main green-citrus data, CARPK, PiXMo-Count, blood-cell counting, and other fruit-counting tasks.

The minimal ablation matrix is:

- one-shot SAM3 using only the user prompt;
- user prompt + tiled bootstrap;
- fixed manually generated global prompt bank;
- Qwen prompt bank without feedback/replanning;
- full Qwen replanning;
- positive prompts only;
- positive + confounder prompts;
- global vectorized sensing versus per-node verification;
- hard threshold counting versus posterior soft counting;
- with/without semantic deduplication of paraphrases;
- with/without targeted residual cleanup.

Report target precision/recall/F1, count MAE or absolute error, discovery recall, duplicate/merge errors, calibration where meaningful, number of SAM3 executions, number of Qwen calls, wall-clock time, GPU memory, and storage. For the main claim, plot performance versus sensing cost, not only performance versus iteration number.

A crucial falsification test is whether adaptive Qwen replanning yields measurable benefit over a single Qwen-generated static bank after controlling for the total number of SAM3 passes. Another is whether negative/confounder prompts specifically improve citrus-versus-leaf separation.

## 14. Limitations

The formulation currently relies on approximate components. Qwen-proposed $\beta$ values are semantic priors rather than calibrated probabilities. Arbitrary prompt-conditioned SAM3 channels $g_x$ are initially unknown. The missed-object state $M_t^\star$ is conceptually important but difficult to estimate zero-shot. Graph registration can introduce count bias through merges and duplicates. Node-level conditional independence is only an approximation, and global prompts can produce correlated failures.

These limitations should remain explicit. V4 is designed so that each approximation is isolated and can be calibrated or replaced without changing the overall active-perception architecture.

## 15. Contribution and positioning

The proposed contribution is not simply "use Qwen to generate more SAM3 prompts." It is the synthesis of:

1. a scene-level active-perception formulation for zero-shot counting;
2. sensor-first bootstrap before language planning;
3. Qwen as an amortized proposer over a combinatorial semantic action space;
4. global vectorized prompt-conditioned sensing, where one SAM3 action updates many objects;
5. target and confounder prompts as symmetric Bayesian experiments;
6. adaptive replanning and sequential stopping under sensing cost;
7. explicit separation between represented-object uncertainty and residual discovery uncertainty.

This framing connects active perception (Bajcsy), active sequential testing (Chernoff; Naghshvar and Javidi), information pursuit, submodular coverage, and prompt-conditioned foundation models in a single zero-shot counting system.

## 16. One-paragraph summary

V4 treats counting as **closed-loop control of a semantic visual sensor**. The user supplies the target concept; SAM3 first reveals what that concept retrieves in the actual image. Qwen then sees the scene and representative sensor outputs and proposes target, confounder, context, and verification experiments. The controller executes the most useful prompts globally, associates their detections with an evolving graph, discovers new objects, and updates competing class beliefs for many objects at once. Qwen replans only when the accumulated observations materially change the problem. Local crop verification is reserved for residual ambiguity. Mathematically, Bajcsy provides the controlled-sensing perspective, ASHT provides posterior experiment selection and stopping, and an additional discovery term handles objects not yet represented in the hypothesis set. The terminal goal is not certainty about every crop but minimum counting risk per unit of computation.

## Working references

The current formulation builds primarily on the references already collected in the V3 mathematical notes: Bajcsy, *Active Perception* (1988); Bajcsy, Aloimonos and Tsotsos, *Revisiting Active Perception* (2018); Chernoff, *Sequential Design of Experiments* (1959); Naghshvar and Javidi, *Active Sequential Hypothesis Testing* (2013) and *Sequentiality and Adaptivity Gains in Active Hypothesis Testing* (2013); Jedynak, Frazier and Sznitman, *Twenty Questions with Noise* (2012); Chattopadhyay et al., *Interpretable by Design / Information Pursuit* and *Variational Information Pursuit*; Nemhauser, Wolsey and Fisher on submodular coverage; and the SAM/SAM3, Qwen-VL, CLIP/CuPL and zero-shot attribute literature listed in the existing project notes. Bibliographic details should be verified against the source PDFs before submission.
