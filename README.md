# Awesome-LLM-for-RecSys [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of AWESOME papers and resources on the large language model (LLM) related recommender system topics. 

:satisfied: Please check out our survey paper for LLM-enhanced RS: [How Can Recommender Systems Benefit from Large Language Models: A Survey](https://arxiv.org/pdf/2306.05817v5.pdf)

To catch up with the latest research progress, this repository will be actively maintained as well as our released survey paper. Newly added papers will first appear in ``1.6 Paper Pending List: to be Added to Our Survey Paper`` section.

:rocket:	**2024.02.05 - Paper v5 released**: New release with 27-page main content & more thorough taxonomies.
<details><summary><b>Survey Paper Update Logs</b></summary>

<p>
<ul>
  <li><b>2023.06.29 - Paper v5 released</b>: New release with 27-page main content & more thorough taxonomies.</li>
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
| LLaRA | LLaRA: Aligning Large Language Models with Sequential Recommenders | LLaMA2 (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.02445) |
| PO4ISR | Large Language Models for Intent-Driven Session Recommendations | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.07552) |
| DRDT | DRDT: Dynamic Reflection with Divergent Thinking for LLM-based Sequential Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.11336) |
| RecPrompt | RecPrompt: A Prompt Tuning Framework for News Recommendation Using Large Language Models | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10463) |
| LiT5 | Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models | T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16098) |
| STELLA | Large Language Models are Not Stable Recommender Systems | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15746) |
| Llama4Rec | Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation | LLaMA2 (7B) | Full Finetuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.13870) |

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

<b>1.4.2 Open-ended User Interaction</b>
    
| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| BARCOR | BARCOR: Towards A Unified Framework for Conversational Recommendation Systems | BART-base (139M) | Selective-layer Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2203.14257) |
| RecInDial | RecInDial: A Unified Framework for Conversational Recommendation with Pretrained Language Models | DialoGPT (110M) | Full Finetuning | AACL 2022 | [[Link]](https://arxiv.org/abs/2110.07477) |
| UniCRS | Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning | DialoGPT-small (176M) | Frozen | KDD 2022 | [[Link]](https://arxiv.org/abs/2206.09363) |
| T5-CR | Multi-Task End-to-End Training Improves Conversational Recommendation | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06218) |
| TtW | Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation | T5-base (223M) & T5-XXL (11B) | Full Finetuning & Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2301.11489) |
| N/A | Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models | ChatGPT | Frozen | EMNLP 2023 | [[Link]](https://arxiv.org/abs/2305.13112) |

</p>
</details>

<details><summary><b>1.5 LLM for RS Pipeline Controller</b></summary>
<p>
    
| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| Chat-REC | Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2303.14524) |
| RecLLM | Leveraging Large Language Models in Conversational Recommender Systems | LLaMA (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07961) |
| RAH | RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.09904) |
| RecMind | RecMind: Large Language Model Powered Agent For Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.14296) |
| InteRecAgent | Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations | GPT4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.16505) |
| CORE | Lending Interaction Wings to Recommender Systems with Conversational Agents | N/A | N/A | NIPS 2023 | [[Link]](https://arxiv.org/abs/2310.04230) |

</p>
</details>

<details><summary><b>1.6 Other Related Papers</b></summary>
<p>

<b>1.6.1 Related Survey Papers</b>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
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

<b>1.6.2 Other Papers</b>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| Large Language Model Can Interpret Latent Space of Sequential Recommender | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.20487) |
| Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.01026) |
| INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.06532) |
| Evaluation of Synthetic Datasets for Conversational Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2212.08167v1) |
| Generative Recommendation: Towards Next-generation Recommender Paradigm | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03516) |
| Towards Personalized Prompt-Model Retrieval for Generative Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.02205) |
| Generative Next-Basket Recommendation | RecSys 2023 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3604915.3608823) |
    
</p>
</details>

<details><summary><b>1.7 Paper Pending List: to be Added to Our Survey Paper</b></summary>
<p>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
|  | A Large Language Model Enhanced Conversational Recommender System |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.06212) |
|  | Improving Conversational Recommendation Systems via Bias Analysis and Language-Model-Enhanced Data Augmentation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.16738) |
|  | Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10967) |
|  | Unlocking the Potential of Large Language Models for Explainable Recommendations |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15661) |
|  | The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15524) |
|  | Empowering Few-Shot Recommender Systems with Large Language Models -- Enhanced Representations |  |  | IEEE Access | [[Link]](https://arxiv.org/abs/2312.13557) |
|  | dIR -- Discrete Information Retrieval: Conversational Search over Unstructured (and Structured) Data with Large Language Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.13264) |
| Logic-Scaffolding | Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs | Falcon (40B) | Frozen | WSDM 2024 | [[Link]](https://arxiv.org/abs/2312.14345) |
|  | Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.04057) |
|  | ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.03605) |
|  | Combining Embedding-Based and Semantic-Based Models for Post-hoc Explanations in Recommender Systems |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.04474) |
|  | Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.10545) |
|  | LLM4Vis: Explainable Visualization Recommendation using ChatGPT |  |  | EMNLP 2023 | [[Link]](https://arxiv.org/abs/2310.07652) |
|  | Parameter-Efficient Conversational Recommender System as a Language Processing Task |  |  | EACL 2024 | [[Link]](https://arxiv.org/abs/2401.14194) |
|  | Data-efficient Fine-tuning for LLM-based Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.17197) |
|  | Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation |  |  | WWW 2024 | [[Link]](https://arxiv.org/abs/2401.14678) |
|  | LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.17723) |
|  | PAP-REC: Personalized Automatic Prompt for Recommendation Language Model |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.00284) |
|  | From PARIS to LE-PARIS: Toward Patent Response Automation with Recommender Systems and Collaborative Large Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.00421) |
|  | Improving Sequential Recommendations with LLMs |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01339) |
|  | A Multi-Agent Conversational Recommender System |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01135) |
|  | TransFR: Transferable Federated Recommendation with Pre-trained Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01124) |
|  | Large Language Model Distilling Medication Recommendation Model |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.02803) |
|  | Uncertainty-Aware Explainable Recommendation with Large Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.03366) |
|  | Natural Language User Profiles for Transparent and Scrutable Recommendations |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.05810) |
|  | Leveraging LLMs for Unsupervised Dense Retriever Ranking |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.04853) |
|  | RA-Rec: An Efficient ID Representation Alignment Framework for LLM-based Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.04527) |
|  | A Multi-Agent Conversational Recommender System |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.01135) |
|  | Fairly Evaluating Large Language Model-based Recommendation Needs Revisit the Cross-Entropy Loss |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.06216) |
|  | SearchAgent: A Lightweight Collaborative Search Agent with Large Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.06360) |
|  | Large Language Model Interaction Simulator for Cold-Start Item Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09176) |
|  | Enhancing ID and Text Fusion via Alternative Training in Session-based Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08921) |
|  | eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08831) |
|  | LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09617) |
|  | LLM-based Federated Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09959) |
|  | Rethinking Large Language Model Architectures for Sequential Recommendations |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.09543) |
|  | Large Language Model with Graph Convolution for Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08859) |
|  | Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.08670) |
|  | Enhancing Recommendation Diversity by Re-ranking with Large Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.11506) |
|  | Are ID Embeddings Necessary? Whitening Pre-trained Text Embeddings for Effective Sequential Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.10602) |
|  | SPAR: Personalized Content-Based Recommendation via Long Engagement Attention |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.10555) |
|  | Cognitive Personalized Search Integrating Large Language Models with an Efficient Memory Mechanism |  |  | WWW 2024 | [[Link]](https://arxiv.org/abs/2402.10548) |
|  | Large Language Models as Data Augmenters for Cold-Start Item Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.11724) |
|  | Explain then Rank: Scale Calibration of Neural Rankers Using Natural Language Explanations from Large Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.12276) |
|  | LLM4SBR: A Lightweight and Effective Framework for Integrating Large Language Models in Session-based Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.13840) |
|  | Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.13750) |
|  | User-LLM: Efficient LLM Contextualization with User Embeddings |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.13598) |
|  | Stealthy Attack on Large Language Model based Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.14836) |
|  | Multi-Agent Collaboration Framework for Recommender Systems |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.15235) |
|  | Item-side Fairness of Large Language Model-based Recommendation System |  |  | WWW 2024 | [[Link]](https://arxiv.org/abs/2402.15215) |
|  | Integrating Large Language Models with Graphical Session-Based Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.16539) |
|  | Language-Based User Profiles for Recommendation |  |  | LLM-IGS@WSDM2024 | [[Link]](https://arxiv.org/abs/2402.15623) |
|  | BASES: Large-scale Web Search User Simulation with Large Language Model based Agents |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.17505) |
|  | Prospect Personalized Recommendation on Large Language Model-based Agent Platform |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18240) |
|  | Sequence-level Semantic Representation Fusion for Recommender Systems |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18166) |
|  | Corpus-Steered Query Expansion with Large Language Models |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2402.18031) |

</p >
</details>

## 2. Datasets & Benchmarks

The datasets & benchmarks for LLM-related RS topics should maintain the original semantic/textual features, instead of anonymous feature IDs.

### 2.1 Datasets

| **Dataset** | **RS Scenario** | **Link** |
|:---:|:---:|:---:|
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
@article{lin2023can,
  title={How Can Recommender Systems Benefit from Large Language Models: A Survey},
  author={Lin, Jianghao and Dai, Xinyi and Xi, Yunjia and Liu, Weiwen and Chen, Bo and Li, Xiangyang and Zhu, Chenxu and Guo, Huifeng and Yu, Yong and Tang, Ruiming and others},
  journal={arXiv preprint arXiv:2306.05817},
  year={2023}
}
```
