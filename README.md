# Awesome-LLM-for-RecSys [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of AWESOME papers and resources on the large language model (LLM) related recommender system topics. 

:tada: Our survey paper has been accepted by **_ACM Transactions on Information Systems (TOIS)_**: [How Can Recommender Systems Benefit from Large Language Models: A Survey](https://dl.acm.org/doi/10.1145/3678004)

:bell: Since our survey paper is archived, we will update the latest research works at ``1.7 Newest Research Work List``.

:grin: I am also wrting weekly paper notes about latest LLM-enhanced RS at WeChat. Welcome to follow by scanning the [QR-Code](https://github.com/CHIANGEL/Awesome-LLM-for-RecSys/blob/main/wechat_for_paper_notes.jpeg).

:rocket:	**2024.07.09 - Paper v6 released**: Our archived camera-ready version for TOIS.
<details><summary><b>Survey Paper Update Logs</b></summary>

<p>
<ul>
  <li><b>2024.07.09 - Paper v6 released</b>: Our camera-ready Version for TOIS, which will be archived.</li>
  <li><b>2024.02.05 - Paper v5 released</b>: New release with 27-page main content & more thorough taxonomies.</li>
  <li><b>2023.06.29 - Paper v4 released</b>: 7 papers have been newly added.</li>
  <li><b>2023.06.28 - Paper v3 released</b>: Fix typos.</li>
  <li><b>2023.06.12 - Paper v2 released</b>: Add summerization table in the appendix.</li>
  <li><b>2023.06.09 - Paper v1 released</b>: Initial version.</li>
</ul>
</p>

</details>

## 1. Papers

We classify papers according to where LLM will be adapted in the pipeline of RS, which is summarized in the figure below.

<img width="650" src="https://github.com/CHIANGEL/Awesome-LLM-for-RecSys/blob/main/where-framework-1.png">

<details><summary><b>1.1 LLM for Feature Engineering</b></summary>
<p>

<b>1.1.1 User- and Item-level Feature Augmentation</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| LLM4KGC | Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs | PaLM (540B)/ ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.09858v1) |
| TagGPT | TagGPT: Large Language Models are Zero-shot Multimodal Taggers | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03022v1) |
| ICPC | Large Language Models for User Interest Journeys | LaMDA (137B) | Full Finetuning/ Prompt Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.15498) |
| KAR | Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.10933) |
| PIE | Product Information Extraction using ChatGPT | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.14921) |
| LGIR | Enhancing Job Recommendation through LLM-based Generative Adversarial Networks | GhatGLM (6B) | Frozen | AAAI 2024 | [[Link]](https://arxiv.org/abs/2307.10747) |
| GIRL | Generative Job Recommendations with Large Language Model | BELLE (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.02157) |
| LLM-Rec | LLM-Rec: Personalized Recommendation via Prompting Large Language Models | text-davinci-003 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.15780) |
| HKFR | Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM | ChatGPT | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2308.03333) |
| LLaMA-E | LLaMA-E: Empowering E-commerce Authoring with Multi-Aspect Instruction Following | LLaMA (30B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.04913) |
| EcomGPT | EcomGPT: Instruction-tuning Large Language Models with Chain-of-Task Tasks for E-commerce | BLOOMZ (7.1B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.06966) |
| TF-DCon | Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09874) |
| RLMRec | Representation Learning with Large Language Models for Recommendation | ChatGPT | Frozen | WWW 2024 | [[Link]](https://arxiv.org/abs/2310.15950) |
| LLMRec | LLMRec: Large Language Models with Graph Augmentation for Recommendation | ChatGPT | Frozen | WSDM 2024 | [[Link]](https://arxiv.org/pdf/2311.00423.pdf) |
| LLMRG | Enhancing Recommender Systems with Large Language Model Reasoning Graphs | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.10835) |
| CUP | Recommendations by Concise User Profiles from Review Text | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01314) |
| SINGLE | Modeling User Viewing Flow using Large Language Models for Article Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.07619) |
| SAGCN | Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via Large Language Models | Vicuna (13B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16275) |
| UEM | User Embedding Model for Personalized Language Prompting | FLAN-T5-base (250M) | Full Finetuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.04858) |
| LLMHG | LLM-Guided Multi-View Hypergraph Learning for Human-Centric Explainable Recommendation | GPT4 | Frozen | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.08217) |
| Llama4Rec | Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation | LLaMA2 (7B) | Full Finetuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.13870) |
| LLM4Vis | LLM4Vis: Explainable Visualization Recommendation using ChatGPT | ChatGPT | Frozen | EMNLP 2023 | [[Link]](https://arxiv.org/abs/2310.07652) |
| LoRec | LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks | LLaMA2 | Frozen | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2401.17723) |

<b>1.1.2 Instance-level Sample Generation</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| GReaT | Language Models are Realistic Tabular Data Generators | GPT2-medium (355M) | Full Finetuning | ICLR 2023 | [[Link]](https://arxiv.org/abs/2210.06280) |
| ONCE | ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models | ChatGPT | Frozen | WSDM 2024 | [[Link]](https://arxiv.org/abs/2305.06566) |
| AnyPredict | AnyPredict: Foundation Model for Tabular Prediction | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.12081) |
| DPLLM | Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models | T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.05973) |
| MINT | Large Language Model Augmented Narrative Driven Recommendations | text-davinci-003 | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2306.02250) |
| Agent4Rec | On Generative Agents in Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.10108) |
| RecPrompt | RecPrompt: A Prompt Tuning Framework for News Recommendation Using Large Language Models | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10463) |
| PO4ISR | Large Language Models for Intent-Driven Session Recommendations | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.07552) |
| BEQUE | Large Language Model based Long-tail Query Rewriting in Taobao Search | ChatGLM (6B) | FFT | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.03758) |
| Agent4Ranking | Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15450) |
| PopNudge | Improving Conversational Recommendation Systems via Bias Analysis and Language-Model-Enhanced Data Augmentation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.16738) |

</p>
</details>

<details><summary><b>1.2 LLM as Feature Encoder</b></summary>
<p>

<b>1.2.1 Representation Enhancement</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| U-BERT | U-BERT: Pre-training User Representations for Improved Recommendation | BERT-base (110M) | Full Finetuning | AAAI 2021 | [[Link]](https://ojs.aaai.org/index.php/AAAI/article/view/16557) |
| UNBERT | UNBERT: User-News Matching BERT for News Recommendation | BERT-base (110M) | Full Finetuning | IJCAI 2021 | [[Link]](https://www.ijcai.org/proceedings/2021/462) |
| PLM-NR | Empowering News Recommendation with Pre-trained Language Models | RoBERTa-base (125M) | Full Finetuning | SIGIR 2021 | [[Link]](https://arxiv.org/abs/2104.07413) |
| Pyramid-ERNIE | Pre-trained Language Model based Ranking in Baidu Search | ERNIE (110M) | Full Finetuning | KDD 2021 | [[Link]](https://arxiv.org/abs/2105.11108) |
| ERNIE-RS | Pre-trained Language Model for Web-scale Retrieval in Baidu Search | ERNIE (110M) | Full Finetuning | KDD 2021 | [[Link]](https://arxiv.org/abs/2106.03373) |
| CTR-BERT | CTR-BERT: Cost-effective knowledge distillation for billion-parameter teacher models | Customized BERT (1.5B) | Full Finetuning | ENLSP 2021 | [[Link]](https://neurips2021-nlp.github.io/papers/20/CameraReady/camera_ready_final.pdf) |
| SuKD | Learning Supplementary NLP Features for CTR Prediction in Sponsored Search | RoBERTa-large (355M) | Full Finetuning | KDD 2022 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3534678.3539064) |
| PREC | Boosting Deep CTR Prediction with a Plug-and-Play Pre-trainer for News Recommendation | BERT-base (110M) | Full Finetuning | COLING 2022 | [[Link]](https://aclanthology.org/2022.coling-1.249/) |
| MM-Rec | MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation | BERT-base (110M) | Full Finetuning | SIGIR 2022 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3477495.3531896) |
| Tiny-NewsRec | Tiny-NewsRec: Effective and Efficient PLM-based News Recommendation | UniLMv2-base (110M) | Full Finetuning | EMNLP 2022 | [[Link]](https://arxiv.org/abs/2112.00944) |
| PLM4Tag | PTM4Tag: Sharpening Tag Recommendation of Stack Overflow Posts with Pre-trained Models | CodeBERT (125M) | Full Finetuning | ICPC 2022 | [[Link]](https://arxiv.org/abs/2203.10965) |
| TwHIN-BERT | TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations | BERT-base (110M) | Full Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2209.07562) |
| LSH | Improving Code Example Recommendations on Informal Documentation Using BERT and Query-Aware LSH: A Comparative Study | BERT-base (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.03017v1) |
| LLM2BERT4Rec | Leveraging Large Language Models for Sequential Recommendation | text-embedding-ada-002 | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2309.09261) | 
| LLM4ARec | Prompt Tuning Large Language Models on Personalized Aspect Extraction for Recommendations | GPT2 (110M) | Prompt Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.01475) |
| TIGER | Recommender Systems with Generative Retrieval | Sentence-T5-base (223M) | Frozen | NIPS 2023 | [[Link]](https://arxiv.org/abs/2305.05065) |
| TBIN | TBIN: Modeling Long Textual Behavior Data for CTR Prediction | BERT-base (110M) | Frozen | DLP-RecSys 2023 | [[Link]](https://arxiv.org/abs/2308.08483) |
| LKPNR | LKPNR: LLM and KG for Personalized News Recommendation Framework | LLaMA2 (7B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.12028) |
| SSNA | Towards Efficient and Effective Adaptation of Large Language Models for Sequential Recommendation | DistilRoBERTa-base (83M) | Layerwise Adapter Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.01612) |
| CollabContext | Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model | Instructor-XL (1.5B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09400) |
| LMIndexer | Language Models As Semantic Indexers | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.07815) |
| Stack | A BERT based Ensemble Approach for Sentiment Classification of Customer Reviews and its Application to Nudge Marketing in e-Commerce | BERT-base (110M) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.10782) |
| N/A | Utilizing Language Models for Tour Itinerary Recommendation | BERT-base (110M) | Full Finetuning | PMAI@IJCAI 2023 | [[Link]](https://arxiv.org/abs/2311.12355) |
| UEM | User Embedding Model for Personalized Language Prompting | Sentence-T5-base (223M) | Frozen | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.04858) |
| Social-LLM | Social-LLM: Modeling User Behavior at Scale using Language Models and Social Network Data | SBERT-MPNet-base (110M) | Frozen | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.00893) |
| LLMRS | LLMRS: Unlocking Potentials of LLM-Based Recommender Systems for Software Purchase | MPNet (110M) | Frozen | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.06676) |
| KERL | Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems | BERT-mini | Frozen | TNNLS | [[Link]](https://arxiv.org/abs/2312.10967) |
| N/A | Empowering Few-Shot Recommender Systems with Large Language Models -- Enhanced Representations | ChatGPT | Frozen | IEEE Access | [[Link]](https://arxiv.org/abs/2312.13557) |
| N/A | Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations | Unknown | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.08121) |

<b>1.2.2 Unified Cross-domain Recommendation</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| ZESRec | Zero-Shot Recommender Systems | BERT-base (110M) | Frozen | Arxiv 2021 | [[Link]](https://arxiv.org/abs/2105.08318) |
| UniSRec | Towards Universal Sequence Representation Learning for Recommender Systems | BERT-base (110M) | Frozen | KDD 2022 | [[Link]](https://arxiv.org/abs/2206.05941) |
| TransRec | TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback | BERT-base (110M) | Full Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2206.06190) |
| VQ-Rec | Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders | BERT-base (110M) | Frozen | WWW 2023 | [[Link]](https://arxiv.org/abs/2210.12316) |
| IDRec vs MoRec | Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited | BERT-base (110M) | Full Finetuning | SIGIR 2023 | [[Link]](https://arxiv.org/abs/2303.13835) |
| TransRec | Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights | RoBERTa-base (125M) | Layerwise Adapter Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.15036) |
| TCF | Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights | OPT-175B (175B) | Frozen/ Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.11700) |
| S&R Foundation | An Unified Search and Recommendation Foundation Model for Cold-Start Scenario | ChatGLM (6B) | Frozen | CIKM 2023 | [[Link]](https://arxiv.org/abs/2309.08939) |
| MISSRec | MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation | CLIP-B/32 (400M) | Full Finetuning | MM 2023 | [[Link]](https://arxiv.org/abs/2308.11175) |
| UFIN | UFIN: Universal Feature Interaction Network for Multi-Domain Click-Through Rate Prediction | FLAN-T5-base (250M) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.15493) |
| PMMRec | Multi-Modality is All You Need for Transferable Recommender Systems | RoBERTa-large (355M) | Top-2-layer Finetuning | ICDE 2024 | [[Link]](https://arxiv.org/abs/2312.09602) |
| Uni-CTR | A Unified Framework for Multi-Domain CTR Prediction via Large Language Models | Sheared-LLaMA (1.3B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10743) |
| PCDR | Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation | BERT-base (110M) | Frozen | WWW 2024 | [[Link]](https://arxiv.org/abs/2401.14678) |

</p>
</details>

<details><summary><b>1.3 LLM as Scoring/Ranking Function</b></summary>
<p>

<b>1.3.1 Item Scoring Task</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| LMRecSys | Language Models as Recommender Systems: Evaluations and Limitations | GPT2-XL (1.5B) | Full Finetuning | ICBINB 2021 | [[Link]](https://openreview.net/forum?id=hFx3fY7-m9b) |
| PTab | PTab: Using the Pre-trained Language Model for Modeling Tabular Data | BERT-base (110M) | Full Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2209.08060) |
| UniTRec | UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation | BART (406M) | Full Finetuning | ACL 2023 | [[Link]](https://arxiv.org/abs/2305.15756) |
| Prompt4NR | Prompt Learning for News Recommendation | BERT-base (110M) | Full Finetuning | SIGIR 2023 | [[Link]](https://arxiv.org/abs/2304.05263) |
| RecFormer | Text Is All You Need: Learning Language Representations for Sequential Recommendation | LongFormer (149M) | Full Finetuning | KDD 2023 | [[Link]](https://arxiv.org/abs/2305.13731v1) |
| TabLLM | TabLLM: Few-shot Classification of Tabular Data with Large Language Models | T0 (11B) | Few-shot Parameter-effiecnt Finetuning | AISTATS 2023 | [[Link]](https://arxiv.org/abs/2210.10723) |
| Zero-shot GPT | Zero-Shot Recommendation as Language Modeling | GPT2-medium (355M) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2112.04184) |
| FLAN-T5 | Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction | FLAN-5-XXL (11B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/pdf/2305.06474.pdf) |
| BookGPT | BookGPT: A General Framework for Book Recommendation Empowered by Large Language Model | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.15673v1) |
| TALLRec | TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation | LLaMA (7B) | LoRA | RecSys 2023 | [[Link]](https://arxiv.org/abs/2305.00447) |
| PBNR | PBNR: Prompt-based News Recommender System | T5-small (60M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.07862) |
| CR-SoRec | CR-SoRec: BERT driven Consistency Regularization for Social Recommendation | BERT-base (110M) | Full Finetuning | RecSys 2023 | [[Link]](https://dl.acm.org/doi/fullHtml/10.1145/3604915.3608844) |
| PromptRec | Towards Personalized Cold-Start Recommendation with Prompts | LLaMA (7B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.17256) |
| GLRec | Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations | BELLE-LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.05722) |
| BERT4CTR | BERT4CTR: An Efficient Framework to Combine Pre-trained Language Model with Non-textual Features for CTR Prediction | RoBERTa-large (355M) | Full Finetuning | KDD 2023 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3580305.3599780) |
| ReLLa | ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation | Vicuna (13B) | LoRA | WWW 2024 | [[Link]](https://arxiv.org/abs/2308.11131) |
| TASTE | Text Matching Improves Sequential Recommendation by Reducing Popularity Biases | T5-base (223M) | Full Finetuning | CIKM 2023 | [[Link]](https://arxiv.org/abs/2308.14029) |
| N/A | Unveiling Challenging Cases in Text-based Recommender Systems | BERT-base (110M) | Full Finetuning | RecSys Workshop 2023 | [[Link]](https://ceur-ws.org/Vol-3476/paper5.pdf) |
| ClickPrompt | ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction | RoBERTa-large (355M) | Full Finetuning | WWW 2024 | [[Link]](https://arxiv.org/abs/2310.09234) |
| SetwiseRank | A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models | FLAN-T5-XXL (11B) | Frozen |  Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09497) |
| UPSR | Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.13540) |
| LLM-Rec | One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems | OPT (6.7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.14304) |
| LLMRanker | Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels | FLAN PaLM2 S | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.14122) |
| CoLLM | CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation | Vicuna (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.19488) |
| FLIP | FLIP: Towards Fine-grained Alignment between ID-based Models and Pretrained Language Models for CTR Prediction | RoBERTa-large (355M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.19453) |
| BTRec | BTRec: BERT-Based Trajectory Recommendation for Personalized Tours | BERT-base (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.19886) |
| CLLM4Rec | Collaborative Large Language Model for Recommender Systems | GPT2 (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01343) |
| CUP | Recommendations by Concise User Profiles from Review Text | BERT-base (110M) | Last-layer Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01314) |
| N/A | Instruction Distillation Makes Large Language Models Efficient Zero-shot Rankers | FLAN-T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01555) |
| CoWPiRec | Collaborative Word-based Pre-trained Item Representation for Transferable Recommendation | BERT-base (110M) | Full Finetuning | ICDM 2023 | [[Link]](https://arxiv.org/abs/2311.10501) |
| RecExplainer | RecExplainer: Aligning Large Language Models for Recommendation Model Interpretability | Vicuna-v1.3 (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.10947) |
| E4SRec | E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation | LLaMA2 (13B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.02443) |
| CER | The Problem of Coherence in Natural Language Explanations of Recommendations | GPT2 (110M) | Full Finetuning | ECAI 2023 | [[Link]](https://arxiv.org/abs/2312.11356) |
| LSAT | Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems | LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15599) |
| Llama4Rec | Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation | LLaMA2 (7B) | Full Finetuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.13870) |
    
<b>1.3.2 Item Generation Task</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| GPT4Rec | GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation | GPT2 (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03879) |
| VIP5 | VIP5: Towards Multimodal Foundation Models for Recommendation | T5-base (223M) | Layerwise Adater Tuning | EMNLP 2023 | [[Link]](https://arxiv.org/abs/2305.14302) |
| P5-ID | How to Index Item IDs for Recommendation Foundation Models | T5-small (60M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06569) |
| FaiRLLM | Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation | ChatGPT | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2305.07609) |
| PALR | PALR: Personalization Aware LLMs for Recommendation | LLaMA (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07622) |
| ChatGPT | Large Language Models are Zero-Shot Rankers for Recommender Systems | ChatGPT | Frozen | ECIR 2024 | [[Link]](https://arxiv.org/abs/2305.08845) |
| AGR | Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.04518) |
| NIR | Zero-Shot Next-Item Recommendation using Large Pretrained Language Models | GPT3 (175B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03153) |
| GPTRec | Generative Sequential Recommendation with GPTRec | GPT2-medium (355M) | Full Finetuning | Gen-IR@SIGIR 2023 | [[Link]](https://arxiv.org/abs/2306.11114) |
| ChatNews | A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.10702) |
| N/A | Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences | PaLM (62B) | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2307.14225) |
| LLMSeqPrompt | Leveraging Large Language Models for Sequential Recommendation | OpenAI ada model | Finetune | RecSys 2023 | [[Link]](https://arxiv.org/abs/2309.09261) | 
| GenRec | GenRec: Large Language Model for Generative Recommendation | LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.00457) |
| UP5 | UP5: Unbiased Foundation Model for Fairness-aware Recommendation | T5-base (223M) | Prefix Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.12090) |
| HKFR | Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM | ChatGLM (6B) | LoRA | RecSys 2023 | [[Link]](https://arxiv.org/abs/2308.03333) |
| N/A | The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendations | ChatGPT | Frozen | EAAMO 2023 | [[Link]](https://arxiv.org/abs/2308.02053) |
| BIGRec | A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems | LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.08434) |
| KP4SR | Knowledge Prompt-tuning for Sequential Recommendation | T5-small (60M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.08459) |
| RecSysLLM | Leveraging Large Language Models for Pre-trained Recommender Systems | GLM (10B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.10837) |
| POD | Prompt Distillation for Efficient LLM-based Recommendation | T5-small (60M) | Full Finetuning | CIKM 2023 | [[Link]](https://lileipisces.github.io/files/CIKM23-POD-paper.pdf) |
| N/A | Evaluating ChatGPT as a Recommender System: A Rigorous Approach | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.03613) |
| RaRS | Retrieval-augmented Recommender System: Enhancing Recommender Systems with Large Language Models | ChatGPT | Frozen | RecSys Doctoral Symposium 2023 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3604915.3608889) |
| JobRecoGPT | JobRecoGPT -- Explainable job recommendations using LLMs | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.11805) |
| LANCER | Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling | GPT2 (110M) | Prefix Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.10435) |
| TransRec | A Multi-facet Paradigm to Bridge Large Language Model and Recommendation | LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.06491) |
| AgentCF | AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems | text-davinci-003 & gpt-3.5-turbo | Frozen | WWW 2024 | [[Link]](https://arxiv.org/abs/2310.09233) |
| P4LM | Factual and Personalized Recommendations using Language Models and Reinforcement Learning | PaLM2-XS | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.06176) |
| InstructMK | Multiple Key-value Strategy in Recommendation Systems Incorporating Large Language Model | LLaMA (7B) | Full Finetuning | CIKM GenRec 2023 | [[Link]](https://arxiv.org/abs/2310.16409) |
| LightLM | LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation | T5-small (60M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.17488) |
| LlamaRec | LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking | LLaMA2 (7B) | QLoRA | PGAI@CIKM 2023 | [[Link]](https://arxiv.org/abs/2311.02089) |
| N/A | Exploring Recommendation Capabilities of GPT-4V(ision): A Preliminary Case Study | GPT-4V | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.04199) |
| N/A | Exploring Fine-tuning ChatGPT for News Recommendation | ChatGPT | gpt-3.5-turbo finetuning API | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.05850) |
| N/A | Do LLMs Implicitly Exhibit User Discrimination in Recommendation? An Empirical Study | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.07054) |
| LC-Rec | Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation | LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.09049) |
| DOKE | Knowledge Plugins: Enhancing Large Language Models for Domain-Specific Recommendations | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.10779) |
| ControlRec | ControlRec: Bridging the Semantic Gap between Language Model and Personalized Recommendation | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.16441) |
| LLaRA | LLaRA: Large Language-Recommendation Assistant | LLaMA2 (7B) | LoRA | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2312.02445) |
| PO4ISR | Large Language Models for Intent-Driven Session Recommendations | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.07552) |
| DRDT | DRDT: Dynamic Reflection with Divergent Thinking for LLM-based Sequential Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.11336) |
| RecPrompt | RecPrompt: A Prompt Tuning Framework for News Recommendation Using Large Language Models | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10463) |
| LiT5 | Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models | T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16098) |
| STELLA | Large Language Models are Not Stable Recommender Systems | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15746) |
| Llama4Rec | Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation | LLaMA2 (7B) | Full Finetuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.13870) |
| RECLLM | Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency | ChatGPT | Frozen | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.10545) |
| DEALRec | Data-efficient Fine-tuning for LLM-based Recommendation | LLaMA (7B) | LoRA | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.17197) |

<b>1.3.3 Hybrid Task</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| P5 | Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5) | T5-base (223M) | Full Finetuning | RecSys 2022 | [[Link]](https://arxiv.org/abs/2203.13366) |
| M6-Rec | M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems | M6-base (300M) | Option Tuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2205.08084) |
| InstructRec | Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach | FLAN-T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07001) |
| ChatGPT | Is ChatGPT a Good Recommender? A Preliminary Study | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.10149) |
| ChatGPT | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.09542) |
| ChatGPT | Uncovering ChatGPT's Capabilities in Recommender Systems | ChatGPT | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2305.02182) |
| BDLM | Bridging the Information Gap Between Domain-Specific Model and General LLM for Personalized Recommendation | Vicuna (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.03778) |
| RecRanker | RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation | LLaMA2 (13B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16018) |

</p>
</details>

<details><summary><b>1.4 LLM for User Interaction</b></summary>
<p>

<b>1.4.1 Task-oriented User Interaction</b>
    
| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| TG-ReDial | Towards Topic-Guided Conversational Recommender System | BERT-base (110M) & GPT2 (110M) | Unknown | COLING 2020 | [[Link]](https://arxiv.org/abs/2010.04125) |
| TCP | Follow Me: Conversation Planning for Target-driven Recommendation Dialogue Systems | BERT-base (110M) | Full Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2208.03516) |
| MESE | Improving Conversational Recommendation Systems' Quality with Context-Aware Item Meta-Information | DistilBERT (67M) & GPT2 (110M) | Full Finetuning | ACL 2022 | [[Link]](https://arxiv.org/abs/2112.08140) |
| UniMIND | A Unified Multi-task Learning Framework for Multi-goal Conversational Recommender Systems | BART-base (139M) | Full Finetuning | ACM TOIS 2023 | [[Link]](https://arxiv.org/abs/2204.06923) |
| VRICR | Variational Reasoning over Incomplete Knowledge Graphs for Conversational Recommendation | BERT-base (110M) | Full Finetuning | WSDM 2023 | [[Link]](https://arxiv.org/abs/2212.11868) |
| KECR | Explicit Knowledge Graph Reasoning for Conversational Recommendation | BERT-base (110M) & GPT2 (110M) | Frozen | ACM TIST 2023 | [[Link]](https://arxiv.org/abs/2305.00783) |
| N/A | Large Language Models as Zero-Shot Conversational Recommenders | GPT4 | Frozen | CIKM 2023 | [[Link]](https://arxiv.org/abs/2308.10053) |
| MuseChat | MuseChat: A Conversational Music Recommendation System for Videos | Vicuna (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.06282) |
| N/A | Conversational Recommender System and Large Language Model Are Made for Each Other in E-commerce Pre-sales Dialogue | Chinese-Alpaca (7B) | LoRA | EMNLP 2023 Findings | [[Link]](https://arxiv.org/abs/2310.14626) |
| N/A | ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback | ChatGPT | Frozen | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.03605) |

<b>1.4.2 Open-ended User Interaction</b>
    
| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| BARCOR | BARCOR: Towards A Unified Framework for Conversational Recommendation Systems | BART-base (139M) | Selective-layer Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2203.14257) |
| RecInDial | RecInDial: A Unified Framework for Conversational Recommendation with Pretrained Language Models | DialoGPT (110M) | Full Finetuning | AACL 2022 | [[Link]](https://arxiv.org/abs/2110.07477) |
| UniCRS | Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning | DialoGPT-small (176M) | Frozen | KDD 2022 | [[Link]](https://arxiv.org/abs/2206.09363) |
| T5-CR | Multi-Task End-to-End Training Improves Conversational Recommendation | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06218) |
| TtW | Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation | T5-base (223M) & T5-XXL (11B) | Full Finetuning & Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2301.11489) |
| N/A | Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models | ChatGPT | Frozen | EMNLP 2023 | [[Link]](https://arxiv.org/abs/2305.13112) |
| PECRS | Parameter-Efficient Conversational Recommender System as a Language Processing Task | GPT2-medium (355M) | LoRA | EACL 2024 | [[Link]](https://arxiv.org/abs/2401.14194) |

</p>
</details>

<details><summary><b>1.5 LLM for RS Pipeline Controller</b></summary>
<p>
    
| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| Chat-REC | Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2303.14524) |
| RecLLM | Leveraging Large Language Models in Conversational Recommender Systems | LLaMA (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07961) |
| RAH | RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.09904) |
| RecMind | RecMind: Large Language Model Powered Agent For Recommendation | ChatGPT | Frozen | NAACL 2024 | [[Link]](https://arxiv.org/abs/2308.14296) |
| InteRecAgent | Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.16505) |
| CORE | Lending Interaction Wings to Recommender Systems with Conversational Agents | N/A | N/A | NIPS 2023 | [[Link]](https://arxiv.org/abs/2310.04230) |
| LLMCRS | A Large Language Model Enhanced Conversational Recommender System | LLaMA (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.06212) |

</p>
</details>

<details><summary><b>1.6 Related Survey Papers</b></summary>
<p>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| Recommender Systems in the Era of Large Language Model Agents: A Survey | Preprint | [[Link]](https://www.researchgate.net/publication/386342676_Recommender_Systems_in_the_Era_of_Large_Language_Model_Agents_A_Survey) |
| A Survey on Efficient Solutions of Large Language Models for Recommendation | Arxiv 2024 | [[Link]](https://www.researchgate.net/publication/385863443_A_Survey_on_Efficient_Solutions_of_Large_Language_Models_for_Recommendation) |
| Towards Next-Generation LLM-based Recommender Systems: A Survey and Beyond | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.19744) |
| Bias and Unfairness in Information Retrieval Systems: New Challenges in the LLM Era | KDD 2024 | [[Link]](https://arxiv.org/abs/2404.11457) |
| All Roads Lead to Rome: Unveiling the Trajectory of Recommender Systems Across the LLM Era | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.10081) |
| Survey for Landing Generative AI in Social and E-commerce Recsys - the Industry Perspectives | Arxiv 2024 | [[Link]](https://www.arxiv.org/abs/2406.06475) |
| A Survey of Generative Search and Recommendation in the Era of Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.16924) |
| When Search Engine Services meet Large Language Models: Visions and Challenges | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.00128) |
| A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys) | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.00579) |
| Exploring the Impact of Large Language Models on Recommender Systems: An Extensive Review | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18590) |
| Foundation Models for Recommender Systems: A Survey and New Perspectives | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.11143) |
| Prompting Large Language Models for Recommender Systems: A Comprehensive Framework and Empirical Analysis | Arixv 2024 | [[Link]](https://arxiv.org/abs/2401.04997) |
| User Modeling in the Era of Large Language Models: Current Research and Future Directions | IEEE Data Engineering Bulletin 2023 | [[Link]](https://arxiv.org/abs/2312.11518) |
| A Survey on Large Language Models for Personalized and Explainable Recommendations | Arxiv 2023 |[[Link]](https://arxiv.org/abs/2311.12338) |
| Large Language Models for Generative Recommendation: A Survey and Visionary Discussions | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.01157) |
| Large Language Models for Information Retrieval: A Survey | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.07107) |
| When Large Language Models Meet Personalization: Perspectives of Challenges and Opportunities | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.16376) | |
| Recommender Systems in the Era of Large Language Models (LLMs) | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.02046) |
| A Survey on Large Language Models for Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.19860) |
| Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems | TACL 2023 | [[Link]](https://arxiv.org/abs/2302.03735) |
| Self-Supervised Learning for Recommender Systems: A Survey | TKDE 2022 | [[Link]](https://arxiv.org/abs/2203.15876) |
    
</p>
</details>

<details><summary><b>1.7 Newest Research Work List</b></summary>
<p>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| Large Language Model Can Interpret Latent Space of Sequential Recommender | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.20487) |
| Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.01026) |
| INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.06532) |
| Evaluation of Synthetic Datasets for Conversational Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2212.08167v1) |
| Generative Recommendation: Towards Next-generation Recommender Paradigm | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03516) |
| Towards Personalized Prompt-Model Retrieval for Generative Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.02205) |
| Generative Next-Basket Recommendation | RecSys 2023 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3604915.3608823) |
| Unlocking the Potential of Large Language Models for Explainable Recommendations | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15661) |
| Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs | Falcon (40B) | Frozen | WSDM 2024 | [[Link]](https://arxiv.org/abs/2312.14345) |
| Improving Sequential Recommendations with LLMs | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01339) |
| A Multi-Agent Conversational Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01135) |
| TransFR: Transferable Federated Recommendation with Pre-trained Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01124) |
| Large Language Model Distilling Medication Recommendation Model | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.02803) |
| Uncertainty-Aware Explainable Recommendation with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.03366) |
| Natural Language User Profiles for Transparent and Scrutable Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.05810) |
| Leveraging LLMs for Unsupervised Dense Retriever Ranking | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.04853) |
| RA-Rec: An Efficient ID Representation Alignment Framework for LLM-based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.04527) |
| A Multi-Agent Conversational Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01135) |
| Fairly Evaluating Large Language Model-based Recommendation Needs Revisit the Cross-Entropy Loss | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.06216) |
| SearchAgent: A Lightweight Collaborative Search Agent with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.06360) |
| Large Language Model Interaction Simulator for Cold-Start Item Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09176) |
| Enhancing ID and Text Fusion via Alternative Training in Session-based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08921) |
| eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08831) |
| LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09617) |
| LLM-based Federated Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09959) |
| Rethinking Large Language Model Architectures for Sequential Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09543) |
| Large Language Model with Graph Convolution for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08859) |
| Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08670) |
| Enhancing Recommendation Diversity by Re-ranking with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.11506) |
| Are ID Embeddings Necessary? Whitening Pre-trained Text Embeddings for Effective Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.10602) |
| SPAR: Personalized Content-Based Recommendation via Long Engagement Attention | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.10555) |
| Cognitive Personalized Search Integrating Large Language Models with an Efficient Memory Mechanism | WWW 2024 | [[Link]](https://arxiv.org/abs/2402.10548) |
| Large Language Models as Data Augmenters for Cold-Start Item Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.11724) |
| Explain then Rank: Scale Calibration of Neural Rankers Using Natural Language Explanations from Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.12276) |
| LLM4SBR: A Lightweight and Effective Framework for Integrating Large Language Models in Session-based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.13840) |
| Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.13750) |
| User-LLM: Efficient LLM Contextualization with User Embeddings | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.13598) |
| Stealthy Attack on Large Language Model based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.14836) |
| Multi-Agent Collaboration Framework for Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.15235) |
| Item-side Fairness of Large Language Model-based Recommendation System | WWW 2024 | [[Link]](https://arxiv.org/abs/2402.15215) |
| Integrating Large Language Models with Graphical Session-Based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.16539) |
| Language-Based User Profiles for Recommendation | LLM-IGS@WSDM2024 | [[Link]](https://arxiv.org/abs/2402.15623) |
| BASES: Large-scale Web Search User Simulation with Large Language Model based Agents | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.17505) |
| Prospect Personalized Recommendation on Large Language Model-based Agent Platform | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18240) |
| Sequence-level Semantic Representation Fusion for Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18166) |
| Corpus-Steered Query Expansion with Large Language Models | ECAL 2024 | [[Link]](https://arxiv.org/abs/2402.18031) |
| NoteLLM: A Retrievable Large Language Model for Note Recommendation | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.01744) |
| An Interpretable Ensemble of Graph and Language Models for Improving Search Relevance in E-Commerce | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.00923) |
| LLM-Ensemble: Optimal Large Language Model Ensemble Method for E-commerce Product Attribute Value Extraction | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.00863) |
| Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.00843) |
| InteraRec: Interactive Recommendations Using Multimodal Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.00822) |
| ChatDiet: Empowering Personalized Nutrition-Oriented Food Recommender Chatbots through an LLM-Augmented Framework  | CHASE 2024 | [[Link]](https://arxiv.org/abs/2403.00781) |
| Towards Efficient and Effective Unlearning of Large Language Models for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.03536) |
| Generative News Recommendation | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.03424) |
| Bridging Language and Items for Retrieval and Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.03952) |
| Can Small Language Models be Good Reasoners for Sequential Recommendation? | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.04260) |
| Aligning Large Language Models for Controllable Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.05063) |
| Personalized Audiobook Recommendations at Spotify Through Graph Neural Networks | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.05185) |
| Towards Graph Foundation Models for Personalization | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.07478) |
| CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.05668) |
| CoRAL: Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.06447) |
| RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems | WWW 2024 Demo | [[Link]](https://arxiv.org/pdf/2403.06465.pdf) |
| KELLMRec: Knowledge-Enhanced Large Language Models for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.06642) |
| USimAgent: Large Language Models for Simulating Search Users | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.09142) |
| CALRec: Contrastive Alignment of Generative LLMs For Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.02429) |
| Integrating Large Language Models with Graphical Session-Based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.16539) |
| Language-Based User Profiles for Recommendation | LLM-IGS@WSDM2024 | [[Link]](https://arxiv.org/abs/2402.15623) |
| BASES: Large-scale Web Search User Simulation with Large Language Model based Agents | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.17505) |
| Prospect Personalized Recommendation on Large Language Model-based Agent Platform | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18240) |
| Sequence-level Semantic Representation Fusion for Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18166) |
| Corpus-Steered Query Expansion with Large Language Models | EACL 2024 | [[Link]](https://arxiv.org/abs/2402.18031) |
| NoteLLM: A Retrievable Large Language Model for Note Recommendation | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.01744) |
| An Interpretable Ensemble of Graph and Language Models for Improving Search Relevance in E-Commerce | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.00923) |
| LLM-Ensemble: Optimal Large Language Model Ensemble Method for E-commerce Product Attribute Value Extraction | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2403.00863) |
| Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2403.00843) |
| Towards Efficient and Effective Unlearning of Large Language Models for Recommendation | FCS | [[Link]](https://arxiv.org/abs/2403.03536) |
| Generative News Recommendation | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.03424) |
| Bridging Language and Items for Retrieval and Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.03952) |
| Can Small Language Models be Good Reasoners for Sequential Recommendation? | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.04260) |
| Aligning Large Language Models for Controllable Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.05063) |
| Personalized Audiobook Recommendations at Spotify Through Graph Neural Networks | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.05185) |
| CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.05668) |
| CoRAL: Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.06447) |
| RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.06465) |
| KELLMRec: Knowledge-Enhanced Large Language Models for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.06642) |
| Towards Graph Foundation Models for Personalization | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.07478) |
| USimAgent: Large Language Models for Simulating Search Users | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.09142) |
| The Whole is Better than the Sum: Using Aggregated Demonstrations in In-Context Learning for Sequential Recommendation | NAACL 2024 | [[Link]](https://arxiv.org/abs/2403.10135) |
| PPM : A Pre-trained Plug-in Model for Click-through Rate Prediction | WWW 2024 | [[Link]](https://arxiv.org/abs/2403.10049) |
| Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.09738) |
| Towards Unified Multi-Modal Personalization: Large Vision-Language Models for Generative Recommendation and Beyond | ICLR 2024 | [[Link]](https://arxiv.org/abs/2403.10667) |
| Harnessing Large Language Models for Text-Rich Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.13325) |
| A Large Language Model Enhanced Sequential Recommender for Joint Video and Comment Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.13574) |
| Could Small Language Models Serve as Recommenders? Towards Data-centric Cold-start Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2306.17256) |
| Play to Your Strengths: Collaborative Intelligence of Conventional Recommender Models and Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.16378) |
| Reinforcement Learning-based Recommender Systems with Large Language Models for State Reward and Action Modeling | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.16948) |
| Large Language Models Enhanced Collaborative Filtering | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.17688) |
| Improving Content Recommendation: Knowledge Graph-Based Semantic Contrastive Learning for Diversity and Cold-Start Users | LREC-COLING 2024 | [[Link]](https://arxiv.org/abs/2403.18667) |
| Sequential Recommendation with Latent Relations based on Large Language Model | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.18348) |
| Enhanced Generative Recommendation via Content and Collaboration Integration | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.18480) |
| To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.18628) |
| IDGenRec: LLM-RecSys Alignment with Textual ID Learning | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2403.19021) |
| Breaking the Length Barrier: LLM-Enhanced CTR Prediction in Long Textual User Behaviors | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2403.19347) |
| Make Large Language Model a Better Ranker | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2403.19181) |
| Do Large Language Models Rank Fairly? An Empirical Study on the Fairness of LLMs as Rankers | NAACL 2024 | [[Link]](https://arxiv.org/abs/2404.03192) |
| IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2404.02059) |
| Where to Move Next: Zero-shot Generalization of LLMs for Next POI Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.01855) |
| Tired of Plugins? Large Language Models Can Be End-To-End Recommender | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.00702) |
| Aligning Large Language Models with Recommendation Knowledge | NAACL 2024 | [[Link]](https://arxiv.org/abs/2404.00245) |
| Enhancing Content-based Recommendation via Large Language Model | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.00236) |
| DRE: Generating Recommendation Explanations by Aligning Large Language Models at Data-level | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.06311) |
| Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.05970) |
| Q-PEFT: Query-dependent Parameter Efficient Fine-tuning for Text Reranking with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.04522) |
| JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.04313) |
| PMG : Personalized Multimodal Generation with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.08677) |
| The Elephant in the Room: Rethinking the Usage of Pre-trained Language Model in Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.08796) |
| Exact and Efficient Unlearning for Large Language Model-based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.10327) |
| Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.11343) |
| Behavior Alignment: A New Perspective of Evaluating LLM-based Conversational Recommendation Systems | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2404.11773) |
| Generating Diverse Criteria On-the-Fly to Improve Point-wise LLM Rankers | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.11960) |
| RecGPT: Generative Personalized Prompts for Sequential Recommendation via ChatGPT Training Paradigm | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.08675) |
| MMGRec: Multimodal Generative Recommendation with Transformer Model | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.16555) |
| Hi-Gen: Generative Retrieval For Large-Scale Personalized E-commerce Search | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.15675) |
| Contrastive Quantization based Semantic Code for Generative Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.14774) |
| ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.15592) |
| Large Language Models for Next Point-of-Interest Recommendation | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2404.17591) |
| Ranked List Truncation for Large Language Model-based Re-Ranking | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2404.18185) |
| Large Language Models as Conversational Movie Recommenders: A User Study | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2404.19093) |
| Distillation Matters: Empowering Sequential Recommenders to Match the Performance of Large Language Model | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.00338) |
| Efficient and Responsible Adaptation of Large Language Models for Robust Top-k Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.00824) |
| FairEvalLLM. A Comprehensive Framework for Benchmarking Fairness in Large Language Model Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.02219) |
| Improve Temporal Awareness of LLMs for Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.02778) |
| CALRec: Contrastive Alignment of Generative LLMs For Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.02429) |
| Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.03988) |
| DynLLM: When Large Language Models Meet Dynamic Graph Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.07580) |
| Learnable Tokenizer for LLM-based Generative Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/2405.07314) |
| CELA: Cost-Efficient Language Model Alignment for CTR Prediction | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.10596) |
| RDRec: Rationale Distillation for LLM-based Recommendation | ACL 2024 | [[Link]](https://arxiv.org/abs/2405.10587) |
| EmbSum: Leveraging the Summarization Capabilities of Large Language Models for Content-Based Recommendations | Arxiv 2024 | [[Link]](https://www.arxiv.org/abs/2405.11441) |
| Reindex-Then-Adapt: Improving Large Language Models for Conversational Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.12119) |
| RecGPT: Generative Pre-training for Text-based Recommendation | ACL 2024 | [[Link]](https://arxiv.org/abs/2405.12715) |
| Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2405.15114) |
| Finetuning Large Language Model for Personalized Ranking | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.16127) |
| LLMs for User Interest Exploration: A Hybrid Approach | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.16363) |
| NoteLLM-2: Multimodal Large Representation Models for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.16789) |
| Multimodality Invariant Learning for Multimedia-Based New Item Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.15783) |
| SLMRec: Empowering Small Language Models for Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.17890) |
| Keyword-driven Retrieval-Augmented Large Language Models for Cold-start User Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.19612) |
| Generating Query Recommendations via LLMs | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.19749) |
| Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2405.20646) |
| DisCo: Towards Harmonious Disentanglement and Collaboration between Tabular and Semantic Space for Recommendation | KDD 2024 | [[Link]](https://arxiv.org/abs/2406.00011) |
| LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.00231) |
| A Practice-Friendly Two-Stage LLM-Enhanced Paradigm in Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.00333) |
| Large Language Models as Recommender Systems: A Study of Popularity Bias | Gen-IR@SIGIR24 | [[Link]](https://arxiv.org/abs/2406.01285) |
| Privacy in LLM-based Recommendation: Recent Advances and Future Directions | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.01363) |
| An LLM-based Recommender System Environment | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.01631) |
| Robust Interaction-based Relevance Modeling for Online E-Commerce and LLM-based Retrieval | ECML-PKDD 2024 | [[Link]](https://arxiv.org/abs/2406.02135) |
| Large Language Models Make Sample-Efficient Recommender Systems | FCS | [[Link]](https://arxiv.org/abs/2406.02368) |
| XRec: Large Language Models for Explainable Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.02377) |
| Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.03085) |
| Large Language Models as Evaluators for Recommendation Explanations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.03248) |
| Text-like Encoding of Collaborative Information in Large Language Models for Recommendation | ACL 2024 | [[Link]](https://arxiv.org/abs/2406.03210) |
| Item-Language Model for Conversational Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.02844) |
| Improving LLMs for Recommendation with Out-Of-Vocabulary Tokens | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.08477) |
| On Softmax Direct Preference Optimization for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.09215) |
| TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.10450) |
| DELRec: Distilling Sequential Pattern to Enhance LLM-based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.11156) |
| TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.11678) |
| Multi-Layer Ranking with Large Language Models for News Source Recommendation | SIGIR 2024 | [[Link]](https://arxiv.org/abs/2406.11745) |
| Intermediate Distillation: Data-Efficient Distillation from Black-Box LLMs for Information Retrieval | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.12169) |
| LLM-enhanced Reranking in Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.12433) |
| LLM4MSR: An LLM-Enhanced Paradigm for Multi-Scenario Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.12529) |
| Taxonomy-Guided Zero-Shot Recommendations with LLMs | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.14043) |
| EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration | KDD 2024 | [[Link]](https://arxiv.org/abs/2406.14017) |
| An Investigation of Prompt Variations for Zero-shot LLM-based Rankers | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.14117) |
| Optimizing Novelty of Top-k Recommendations using Large Language Models and Reinforcement Learning | KDD 2024 | [[Link]](https://arxiv.org/abs/2406.14169) |
| Enhancing Collaborative Semantics of Language Model-Driven Recommendations via Graph-Aware Learning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.13235) |
| Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.14900) |
| FIRST: Faster Improved Listwise Reranking with Single Token Decoding | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.15657) |
| LLM-Powered Explanations: Unraveling Recommendations Through Subgraph Reasoning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.15859) |
| DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.16332) |
| ELCoRec: Enhance Language Understanding with Co-Propagation of Numerical and Categorical Features for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2406.18825) |
| Generative Explore-Exploit: Training-free Optimization of Generative Recommender Systems using LLM Optimizers | ACL 2024 | [[Link]](https://arxiv.org/abs/2406.05255) |
| ProductAgent: Benchmarking Conversational Product Search Agent with Asking Clarification Questions | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.00942) |
| MemoCRS: Memory-enhanced Sequential Conversational Recommender Systems with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.04960) |
| Preference Distillation for Personalized Generative Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.05033) |
| Towards Bridging the Cross-modal Semantic Gap for Multi-modal Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.05420) |
| Language Models Encode Collaborative Signals in Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.05441) |
| A Neural Matrix Decomposition Recommender System Model based on the Multimodal Large Language Model | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.08942) |
| LLMGR: Large Language Model-based Generative Retrieval in Alipay Search | SIGIR 2024 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3626772.3661364) |
| Enhancing Sequential Recommenders with Augmented Knowledge from Aligned Large Language Models | SIGIR 2024 | [[Link]](https://dl.acm.org/doi/10.1145/3626772.3657782) |
| Reinforced Prompt Personalization for Recommendation with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.17115) |
| Improving Retrieval in Sponsored Search by Leveraging Query Context Signals | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.14346) |
| Generative Retrieval with Preference Optimization for E-commerce Search | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.19829) |
| GenRec: Generative Personalized Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.21191v1) |
| Breaking the Hourglass Phenomenon of Residual Quantization: Enhancing the Upper Bound of Generative Retrieval | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2407.21488) |
| Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights | CIKM 2024 | [[Link]](https://arxiv.org/abs/2407.19467) |
| Leveraging LLM Reasoning Enhances Personalized Recommender Systems | ACL 2024 |[[Link]](https://arxiv.org/abs/2408.00802) |
| Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.00878) |
| Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation | Arxiv 2024 | [[Link]](https://www.arxiv.org/abs/2408.03533) |
| Exploring Query Understanding for Amazon Product Search | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.02215) |
| A Decoding Acceleration Framework for Industrial Deployable LLM-based Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.05676) |
| Prompt Tuning as User Inherent Profile Inference Machine | Arxiv 2024 | [[Link]](https://arxiv.org/pdf/2408.06577) |
| Beyond Inter-Item Relations: Dynamic Adaptive Mixture-of-Experts for LLM-Based Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/pdf/2408.07427) |
| Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/pdf/2408.06276) |
| DaRec: A Disentangled Alignment Framework for Large Language Model and Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/pdf/2408.08231) |
| LLM4DSR: Leveraing Large Language Model for Denoising Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/pdf/2408.08208) |
| EasyRec: Simple yet Effective Language Models for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.08821) |
| Collaborative Cross-modal Fusion with Large Language Model for Recommendation | CIKM 2024 | [[Link]](https://www.arxiv.org/abs/2408.08564) |
| Customizing Language Models with Instance-wise LoRA for Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.10159) |
| Efficient and Deployable Knowledge Infusion for Open-World Recommendations via Large Language Models | Arxiv 2024 | [[Link]](https://www.arxiv.org/abs/2408.10520) |
| CoRA: Collaborative Information Perception by Large Language Model's Weights for Recommendation | Arxiv 2024 | [[Link]](https://www.arxiv.org/abs/2408.10645) |
| GANPrompt: Enhancing Robustness in LLM-Based Recommendations with GAN-Enhanced Diversity Prompts | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.09671) |
| Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.09698) |
| DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems | Arxiv | [[Link]](https://arxiv.org/abs/2408.12470) |
| LARR: Large Language Model Aided Real-time Scene Recommendation with Semantic Understanding | RecSys 2024 | [[Link]](https://arxiv.org/abs/2408.11523) |
| SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.08686) |
| Are LLM-based Recommenders Already the Best? Simple Scaled Cross-entropy Unleashes the Potential of Traditional Sequential Recommenders | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.14238) |
| HRGraph: Leveraging LLMs for HR Data Knowledge Graphs with Information Propagation-based Job Recommendation | KaLLM 2024 | [[Link]](https://arxiv.org/abs/2408.13521) |
| An Extremely Data-efficient and Generative LLM-based Reinforcement Learning Agent for Recommenders | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2408.16032) |
| CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent | KDD 2024 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3637528.3671837) |
| Laser: Parameter-Efficient LLM Bi-Tuning for Sequential Recommendation with Collaborative Information | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.01605) |
| MARS: Matching Attribute-aware Representations for Text-based Sequential Recommendation | CIKM 2024 | [[Link]](https://arxiv.org/abs/2409.00702) |
| End-to-End Learnable Item Tokenization for Generative Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.05546) |
| Incorporate LLMs with Influential Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.04827) |
| Enhancing Sequential Recommendations through Multi-Perspective Reflections and Iteration | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.06377) |
| STORE: Streamlining Semantic Tokenization and Generative Recommendation with A Single LLM | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.07276) |
| Multilingual Prompts in LLM-Based Recommenders: Performance Across Languages | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.07604) |
| Unleash LLMs Potential for Recommendation by Coordinating Twin-Tower Dynamic Semantic Token Generator | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.09253) |
| Large Language Model Enhanced Hard Sample Identification for Denoising Recommendation | Arxiv 2024 | [[Link]](https://www.arxiv.org/abs/2409.10343) |
| Chain-of-thought prompting empowered generative user modeling for personalized recommendation | Neural Computing and Applications | [[Link]](https://link.springer.com/article/10.1007/s00521-024-10364-2) |
| Challenging Fairness: A Comprehensive Exploration of Bias in LLM-Based Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.10825) |
| Decoding Style: Efficient Fine-Tuning of LLMs for Image-Guided Outfit Recommendation with Preference | CIKM 2024 | [[Link]](https://arxiv.org/abs/2409.12150) |
| LLM-Powered Text Simulation Attack Against ID-Free Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.11690) |
| FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancement | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.11699) |
| Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation | Arxiv 2024 | [[Link]](http://arxiv.org/abs/2409.11860) |
| HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2409.12740) |
| Large Language Model Ranker with Graph Reasoning for Zero-Shot Recommendation | ICANN 2024 | [[Link]](https://link.springer.com/chapter/10.1007/978-3-031-72344-5_24) |
| User Knowledge Prompt for Sequential Recommendation | RecSys 2024 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3640457.3691714) |
| RLRF4Rec: Reinforcement Learning from Recsys Feedback for Enhanced Recommendation Reranking | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.05939) |
| FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.04927) |
| TLRec: A Transfer Learning Framework to Enhance Large Language Models for Sequential Recommendation Tasks | RecSys 2024 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3640457.3691710) |
| SeCor: Aligning Semantic and Collaborative Representations by Large Language Models for Next-Point-of-Interest Recommendations | RecSys 2024 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3640457.3688124) |
| Efficient Inference for Large Language Model-based Generative Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.05165) |
| Instructing and Prompting Large Language Models for Explainable Cross-domain Recommendations | RecSys 2024 | [[Link]](https://dl.acm.org/doi/10.1145/3640457.3688137) |
| ReLand: Integrating Large Language Models' Insights into Industrial Recommenders via a Controllable Reasoning Pool | RecSys 2024 | [[Link]](https://dl.acm.org/doi/10.1145/3640457.3688131) |
| Inductive Generative Recommendation via Retrieval-based Speculation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.02939) |
| Constructing and Masking Preference Profile with LLMs for Filtering Discomforting Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.05411) |
| Towards Scalable Semantic Representation for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.09560) |
| Large Language Models as Narrative-Driven Recommenders | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.13604) |
| The Moral Case for Using Language Model Agents for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.12123) |
| RosePO: Aligning LLM-based Recommenders with Human Values | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.12519) |
| Comprehending Knowledge Graphs with Large Language Models for Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.12229) |
| Triple Modality Fusion: Aligning Visual, Textual, and Graph Data with Large Language Models for Multi-Behavior Recommendations | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.12228) |
| Improving Pinterest Search Relevance Using Large Language Models | CIKM 2024 Workshop | [[Link]](https://arxiv.org/abs/2410.17152) |
| STAR: A Simple Training-free Approach for Recommendations using Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.16458) |
| End-to-end Training for Recommendation with Language-based User Profiles | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.18870) |
| Knowledge Graph Enhanced Language Agents for Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.19627) |
| Collaborative Knowledge Fusion: A Novel Approach for Multi-task Recommender Systems via LLMs | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.20642) |
| Real-Time Personalization for LLM-based Recommendation with Customized In-Context Learning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.23136) |
| ReasoningRec: Bridging Personalized Recommendations and Human-Interpretable Explanations through LLM Reasoning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2410.23180) |
| Beyond Utility: Evaluating LLM as Recommender | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.00331) |
| Enhancing ID-based Recommendation with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.02041) |
| LLM4PR: Improving Post-Ranking in Search Engine with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.01178) |
| Proactive Detection and Calibration of Seasonal Advertisements with Multimodal Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.00780) |
| Enhancing ID-based Recommendation with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.02041) |
| Transferable Sequential Recommendation via Vector Quantized Meta Learning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.01785) |
| Self-Calibrated Listwise Reranking with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.04602) |
| Enhancing Large Language Model Based Sequential Recommender Systems with Pseudo Labels Reconstruction | ACL Findings 2024 | [[Link]](https://aclanthology.org/2024.findings-emnlp.423/) |
| Unleashing the Power of Large Language Models for Group POI Recommendations | Avrxi 2024 | [[Link]](https://arxiv.org/abs/2411.13415) |
| Scaling Laws for Online Advertisement Retrieval | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.13322) |
| Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.13045) |
| GOT4Rec: Graph of Thoughts for Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.14922) |
| HARec: Hyperbolic Graph-LLM Alignment for Exploration and Exploitation in Recommender Systems | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.13865) |
| Cross-Domain Recommendation Meets Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.19862) |
| Explainable CTR Prediction via LLM Reasoning | WSDM 2025 | [[Link]](https://arxiv.org/abs/2412.02588) |
| Enabling Explainable Recommendation in E-commerce with LLM-powered Product Knowledge Graph | IJCAI Workshop 2025 | [[Link]](https://arxiv.org/abs/2412.01837) |
| Break the ID-Language Barrier: An Adaption Framework for Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.18262) |
| LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2411.13789) |
| Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2412.04107) |
| ULMRec: User-centric Large Language Model for Sequential Recommendation | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2412.05543) |

</p >
</details>

## 2. Datasets & Benchmarks

The datasets & benchmarks for LLM-related RS topics should maintain the original semantic/textual features, instead of anonymous feature IDs.

### 2.1 Datasets

| **Dataset** | **RS Scenario** | **Link** |
|:---:|:---:|:---:|
| AmazonQAC | Query Autocomplete | [[Link]](https://arxiv.org/abs/2411.04129) |
| NineRec | 9 Domains | [[Link]](https://github.com/westlake-repl/NineRec) |
| MicroLens | Video Streaming | [[Link]](https://github.com/westlake-repl/MicroLens?tab=readme-ov-file) |
| Amazon-Review 2023 | E-commerce | [[Link]](https://arxiv.org/abs/2403.03952) |
| Reddit-Movie | Conversational & Movie | [[Link]](https://github.com/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys#large-language-models-as-zero-shot-conversational-recommenders) |
| Amazon-M2 | E-commerce | [[Link]](https://arxiv.org/abs/2307.09688) |
| MovieLens | Movie | [[Link]](https://grouplens.org/datasets/movielens/1m/) |
| Amazon | E-commerce | [[Link]](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) |
| BookCrossing | Book | [[Link]](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) |
| GoodReads | Book | [[Link]](https://mengtingwan.github.io/data/goodreads.html) |
| Anime | Anime | [[Link]](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) |
| PixelRec | Short Video | [[Link]](https://github.com/westlake-repl/PixelRec) |
| Netflix | Movie | [[Link]](https://github.com/HKUDS/LLMRec) |
    
### 2.2 Benchmarks

| **Benchmarks** | **Webcite Link** | **Paper** |
|:---:|:---:|:---:|
| Shopping MMLU | [[Paper]](https://arxiv.org/abs/2410.20745?) |
| Amazon-M2 (KDD Cup 2023) | [[Link]](https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge) | [[Paper]](https://arxiv.org/abs/2307.09688) |
| LLMRec | [[Link]](https://github.com/williamliujl/LLMRec) | [[Paper]](https://arxiv.org/abs/2308.12241) |
| OpenP5 | [[Link]](https://github.com/agiresearch/OpenP5) | [[Paper]](https://arxiv.org/abs/2306.11134) |
| TABLET | [[Link]](https://dylanslacks.website/Tablet) | [[Paper]](https://arxiv.org/abs/2304.13188) |

## 3. Related Repositories

| **Repo Name** | **Maintainer** |
|:---:|:---:|
| [rs-llm-paper-list](https://github.com/wwliu555/rs-llm-paper-list) | [wwliu555](https://github.com/wwliu555) |
| [awesome-recommend-system-pretraining-papers](https://github.com/archersama/awesome-recommend-system-pretraining-papers) | [archersama](https://github.com/archersama) |
| [LLM4Rec](https://github.com/WLiK/LLM4Rec) | [WLiK](https://github.com/WLiK) |
| [Awesome-LLM4RS-Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers) | [nancheng58](https://github.com/nancheng58) |
| [LLM4IR-Survey](https://github.com/RUC-NLPIR/LLM4IR-Survey) | [RUC-NLPIR](https://github.com/RUC-NLPIR) |

## Contributing
👍 Welcome to contribute to this repository.

If you have come across relevant resources or found some errors in this repesitory, feel free to open an issue or submit a pull request.

**Contact**: chiangel [DOT] ljh [AT] gmail [DOT] com

## Citation

```
@article{10.1145/3678004,
author = {Lin, Jianghao and Dai, Xinyi and Xi, Yunjia and Liu, Weiwen and Chen, Bo and Zhang, Hao and Liu, Yong and Wu, Chuhan and Li, Xiangyang and Zhu, Chenxu and Guo, Huifeng and Yu, Yong and Tang, Ruiming and Zhang, Weinan},
title = {How Can Recommender Systems Benefit from Large Language Models: A Survey},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3678004},
doi = {10.1145/3678004},
journal = {ACM Trans. Inf. Syst.},
month = {jul}
}
```
