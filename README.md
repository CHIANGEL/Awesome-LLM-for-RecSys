# Awesome-LLM-for-RecSys [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of AWESOME papers and resources on the large language model (LLM) related recommender system topics. 

:satisfied: Please check out our survey paper for LLM-enhanced RS: [How Can Recommender Systems Benefit from Large Language Models: A Survey](https://arxiv.org/abs/2306.05817)

To catch up with the latest research progress, this repository will be actively maintained as well as our released survey paper. Newly added papers will first appear in ``1.6 Paper Pending List: to be Added to Our Survey Paper`` section.

:rocket:	**2023.06.29 - Paper v4 released**: 7 papers have been newly added.
<details><summary><b>Survey Paper Update Logs</b></summary>

<p>
<ul>
  <li><b>2023.06.29 - Paper v4 released</b>: 7 papers have been newly added.</li>
  <li><b>2023.06.28 - Paper v3 released</b>: Fix typos.</li>
  <li><b>2023.06.12 - Paper v2 released</b>: Add summerization table in the appendix.</li>
  <li><b>2023.06.09 - Paper v1 released</b>: Initial version.</li>
</ul>
</p>

</details>

## 1. Papers

We classify papers according to where LLM will be adapted in the pipeline of RS, which is summarized in the figure below.

<img width="350" src="https://github.com/CHIANGEL/Awesome-LLM-for-RecSys/blob/main/where-framework-1.png">

<details><summary><b>1.1 LLM for Feature Engineering</b></summary>
<p>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| GReaT | Language Models are Realistic Tabular Data Generators | GPT2-medium (355M) | Full Finetuning | ICLR 2023 | [[Link]](https://arxiv.org/abs/2210.06280) |
| ONCE | A First Look at LLM-Powered Generative News Recommendation | ChatGPT | Frozen | WSDM 2024 | [[Link]](https://arxiv.org/abs/2305.06566) |
| AnyPredict | AnyPredict: Foundation Model for Tabular Prediction | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.12081) |
| LLM4KGC | Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs | PaLM (540B)/ ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.09858v1) |
| TagGPT | TagGPT: Large Language Models are Zero-shot Multimodal Taggers | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03022v1) |
| ICPC | Large Language Models for User Interest Journeys | LaMDA (137B) | Full Finetuning/ Prompt Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.15498) |
| DPLLM | Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models | T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.05973) |
| KAR | Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.10933) |
| MINT | Large Language Model Augmented Narrative Driven Recommendations | GPT3 (175B) | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2306.02250) |
| PIE | Product Information Extraction using ChatGPT | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.14921) |
| LGIR | Enhancing Job Recommendation through LLM-based Generative Adversarial Networks | GhatGLM (6B) | Frozen | AAAI 2024 | [[Link]](https://arxiv.org/abs/2307.10747) |
| GIRL | Generative Job Recommendations with Large Language Model | Full Finetuning | BELLE (7B) | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.02157) |
| LLM-Rec | LLM-Rec: Personalized Recommendation via Prompting Large Language Models | text-davinci-003 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.15780) |
| LLaMA-E | LLaMA-E: Empowering E-commerce Authoring with Multi-Aspect Instruction Following | LLaMA (30B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.04913) |
| Agent4Rec | On Generative Agents in Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.10108) |
| TF-DCon | Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09874) |
| RLMRec | Representation Learning with Large Language Models for Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.15950) |
| LLMRec | LLMRec: Large Language Models with Graph Augmentation for Recommendation | ChatGPT | Frozen | WSDM 2024 | [[Link]](https://arxiv.org/pdf/2311.00423.pdf) |
| LLMRG | Enhancing Recommender Systems with Large Language Model Reasoning Graphs | GPT-4 | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.10835) |
| CUP | Recommendations by Concise User Profiles from Review Text | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01314) |
| SINGLE | Modeling User Viewing Flow using Large Language Models for Article Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.07619) |

</p>
</details>

<details><summary><b>1.2 LLM as Feature Encoder</b></summary>
<p>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| U-BERT | U-BERT: Pre-training User Representations for Improved Recommendation | BERT-base (110M) | Full Finetuning | AAAI 2021 | [[Link]](https://ojs.aaai.org/index.php/AAAI/article/view/16557) |
| UNBERT | UNBERT: User-News Matching BERT for News Recommendation | BERT-base (110M) | Full Finetuning | IJCAI 2021 | [[Link]](https://www.ijcai.org/proceedings/2021/462) |
| PLM-NR | Empowering News Recommendation with Pre-trained Language Models | RoBERTa-base (125M) | Full Finetuning | SIGIR 2021 | [[Link]](https://arxiv.org/abs/2104.07413) |
| Pyramid-ERNIE | Pre-trained Language Model based Ranking in Baidu Search | ERNIE (110M) | Full Finetuning | KDD 2021 | [[Link]](https://arxiv.org/abs/2105.11108) |
| ERNIE-RS | Pre-trained Language Model for Web-scale Retrieval in Baidu Search | ERNIE (110M) | Full Finetuning | KDD 2021 | [[Link]](https://arxiv.org/abs/2106.03373) |
| CTR-BERT | CTR-BERT: Cost-effective knowledge distillation for billion-parameter teacher models | Customized BERT (1.5B) | Full Finetuning | ENLSP 2021 | [[Link]](https://neurips2021-nlp.github.io/papers/20/CameraReady/camera_ready_final.pdf) |
| ZESRec | Zero-Shot Recommender Systems | BERT-base (110M) | Frozen | Arxiv 2021 | [[Link]](https://arxiv.org/abs/2105.08318) |
| UniSRec | Towards Universal Sequence Representation Learning for Recommender Systems | BERT-base (110M) | Frozen | KDD 2022 | [[Link]](https://arxiv.org/abs/2206.05941) |
| SuKD | Learning Supplementary NLP Features for CTR Prediction in Sponsored Search | RoBERTa-large (355M) | Full Finetuning | KDD 2022 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3534678.3539064) |
| PREC | Boosting Deep CTR Prediction with a Plug-and-Play Pre-trainer for News Recommendation | BERT-base (110M) | Full Finetuning | COLING 2022 | [[Link]](https://aclanthology.org/2022.coling-1.249/) |
| MM-Rec | MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation | BERT-base (110M) | Full Finetuning | SIGIR 2022 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3477495.3531896) |
| Tiny-NewsRec | Tiny-NewsRec: Effective and Efficient PLM-based News Recommendation | UniLMv2-base (110M) | Full Finetuning | EMNLP 2022 | [[Link]](https://arxiv.org/abs/2112.00944) |
| PLM4Tag | PTM4Tag: Sharpening Tag Recommendation of Stack Overflow Posts with Pre-trained Models | CodeBERT (125M) | Full Finetuning | ICPC 2022 | [[Link]](https://arxiv.org/abs/2203.10965) |
| TwHIN-BERT | TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations | BERT-base (110M) | Full Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2209.07562) |
| TransRec | TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback | BERT-base (110M) | Full Finetuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2206.06190) |
| VQ-Rec | Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders | BERT-base (110M) | Frozen | WWW 2023 | [[Link]](https://arxiv.org/abs/2210.12316) |
| IDRec vs MoRec | Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited | BERT-base (110M) | Full Finetuning | SIGIR 2023 | [[Link]](https://arxiv.org/abs/2303.13835) |
| TransRec | Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights | RoBERTa-base (125M) | Layerwise Adapter Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.15036) |
| LSH | Improving Code Example Recommendations on Informal Documentation Using BERT and Query-Aware LSH: A Comparative Study | BERT-base (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.03017v1) |
| TCF | Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights | OPT-175B (175B) | Frozen/ Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.11700) |
| LLM4ARec | Prompt Tuning Large Language Models on Personalized Aspect Extraction for Recommendations | GPT2 (110M) | Prompt Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.01475) |
| TIGER | Recommender Systems with Generative Retrieval |  |  | NIPS 2023 | [[Link]](https://arxiv.org/abs/2305.05065) |
| TBIN | TBIN: Modeling Long Textual Behavior Data for CTR Prediction | BERT-base (110M) | Frozen | DLP-RecSys 2023 | [[Link]](https://arxiv.org/abs/2308.08483) |
| LKPNR | LKPNR: LLM and KG for Personalized News Recommendation Framework | LLaMA2 (7B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.12028) |
| SSNA | Towards Efficient and Effective Adaptation of Large Language Models for Sequential Recommendation | DistilRoBERTa-base (83M) | Layerwise Adapter Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.01612) |
| S&R Foundation | An Unified Search and Recommendation Foundation Model for Cold-Start Scenario | ChatGLM (6B) | Frozen | CIKM 2023 | [[Link]](https://arxiv.org/abs/2309.08939) |
| CollabContext | Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model | Instructor-XL (1.5B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09400) |
| MISSRec | MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation | CLIP-B/32 (400M) | Full Finetuning | MM 2023 | [[Link]](https://arxiv.org/abs/2308.11175) |
| Stack | A BERT based Ensemble Approach for Sentiment Classification of Customer Reviews and its Application to Nudge Marketing in e-Commerce | BERT-base (110M) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.10782) |
| N/A | Utilizing Language Models for Tour Itinerary Recommendation | BERT-base (110M) | Full Finetuning | PMAI@IJCAI 2023 | [[Link]](https://arxiv.org/abs/2311.12355) |
| UFIN | UFIN: Universal Feature Interaction Network for Multi-Domain Click-Through Rate Prediction | FLAN-T5-base (250M) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.15493) |

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
| ReLLa | ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation | Vicuna (13B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.11131) |
| TASTE | Text Matching Improves Sequential Recommendation by Reducing Popularity Biases | T5-base (223M) | Full Finetuning | CIKM 2023 | [[Link]](https://arxiv.org/abs/2308.14029) |
| N/A | Unveiling Challenging Cases in Text-based Recommender Systems | BERT-base (110M) | Full Finetuning | RecSys Workshop 2023 | [[Link]](https://ceur-ws.org/Vol-3476/paper5.pdf) |
| ClickPrompt | ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction | RoBERTa-large (355M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09234) |
| UPSR | Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.13540) |
| LLM-Rec | One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems | OPT (6.7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.14304) |
| LLMRanker | Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels | FLAN PaLM2 S | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.14122) |
| CoLLM | CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation | Vicuna (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.19488) |
| FLIP | FLIP: Towards Fine-grained Alignment between ID-based Models and Pretrained Language Models for CTR Prediction | RoBERTa-large (355M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.19453) |
| CLLM4Rec | Collaborative Large Language Model for Recommender Systems | GPT2 (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01343) |
| CUP | Recommendations by Concise User Profiles from Review Text | BERT-base (110M) | Last-layer Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01314) |
| N/A | Instruction Distillation Makes Large Language Models Efficient Zero-shot Rankers | FLAN-T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.01555) |
| CoWPiRec | Collaborative Word-based Pre-trained Item Representation for Transferable Recommendation | BERT-base (110M) | Full Finetuning | ICDM 2023 | [[Link]](https://arxiv.org/abs/2311.10501) |
| E4SRec | E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation | LLaMA2 (13B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.02443) |
    
<b>1.3.2 Item Generation Task</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| GPT4Rec | GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation | GPT2 (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03879) |
| UP5 | UP5: Unbiased Foundation Model for Fairness-aware Recommendation | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.12090) |
| VIP5 | VIP5: Towards Multimodal Foundation Models for Recommendation | T5-base (223M) | Layerwise Adater Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.14302) |
| P5-ID | How to Index Item IDs for Recommendation Foundation Models | T5-small (61M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06569) |
| FaiRLLM | Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation | ChatGPT | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2305.07609) |
| PALR | PALR: Personalization Aware LLMs for Recommendation | LLaMA (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07622) |
| ChatGPT | Large Language Models are Zero-Shot Rankers for Recommender Systems | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.08845) |
| AGR | Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.04518) |
| NIR | Zero-Shot Next-Item Recommendation using Large Pretrained Language Models | GPT3 (175B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03153) |
| GPTRec | Generative Sequential Recommendation with GPTRec | GPT2-medium (355M) | Full Finetuning | Gen-IR@SIGIR 2023 | [[Link]](https://arxiv.org/abs/2306.11114) |
| ChatNews | A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.10702) |

<b>1.3.3 Hybrid Task</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| P5 | Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5) | T5-base (223M) | Full Finetuning | RecSys 2022 | [[Link]](https://arxiv.org/abs/2203.13366) |
| M6-Rec | M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems | M6-base (300M) | Option Tuning | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2205.08084) |
| InstructRec | Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach | FLAN-T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07001) |
| ChatGPT | Is ChatGPT a Good Recommender? A Preliminary Study | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.10149) |
| ChatGPT | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.09542) |
| ChatGPT | Uncovering ChatGPT's Capabilities in Recommender Systems | ChatGPT | Frozen | RecSys 2023 | [[Link]](https://arxiv.org/abs/2305.02182) |

</p>
</details>

<details><summary><b>1.4 LLM for RS Pipeline Controller</b></summary>
<p>
    
| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| Chat-REC | Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2303.14524) |
| RecLLM | Leveraging Large Language Models in Conversational Recommender Systems | LLaMA (7B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07961) |

</p>
</details>

<details><summary><b>1.5 Other Related Papers</b></summary>
<p>

<b>1.5.1 Related Survey Papers</b>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
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

<b>1.5.2 Other Papers</b>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.06532) |
| Evaluation of Synthetic Datasets for Conversational Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2212.08167v1) |
| Generative Recommendation: Towards Next-generation Recommender Paradigm | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03516) |
| Towards Personalized Prompt-Model Retrieval for Generative Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.02205) |
| Generative Next-Basket Recommendation | RecSys 2023 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3604915.3608823) |
    
</p>
</details>

<details><summary><b>1.6 Paper Pending List: to be Added to Our Survey Paper</b></summary>
<p>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
|  | Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences |  |  | RecSys 2023 | [[Link]](https://recsys.acm.org/recsys23/accepted-contributions/#content-tab-1-1-tab) |
|  | Leveraging Large Language Models for Sequential Recommendation |  |  | RecSys 2023 | [[Link]](https://arxiv.org/abs/2309.09261) | 
| GenRec | GenRec: Large Language Model for Generative Recommendation | LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2307.00457) |
|  | Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM |  |  | RecSys 2023 | [[Link]](https://arxiv.org/abs/2308.03333) |
|  | A Large Language Model Enhanced Conversational Recommender System |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.06212) |
|  | The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendations |  |  | EAAMO 2023 | [[Link]](https://arxiv.org/abs/2308.02053) |
|  | A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.08434) |
|  | Knowledge Prompt-tuning for Sequential Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.08459) |
|  | Leveraging Large Language Models for Pre-trained Recommender Systems |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.10837) |
|  | Large Language Models as Zero-Shot Conversational Recommenders |  |  | CIKM 2023 | [[Link]](https://arxiv.org/abs/2308.10053) |
|  | RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.09904) |
|  | LLMRec: Benchmarking Large Language Models on Recommendation Task |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.12241) |
|  | Prompt Distillation for Efficient LLM-based Recommendation |  |  | CIKM 2023 | [[Link]](https://lileipisces.github.io/files/CIKM23-POD-paper.pdf) |
|  | RecMind: Large Language Model Powered Agent For Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.14296) |
|  | Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.01026) |
|  | Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.16505) |
|  | Evaluating ChatGPT as a Recommender System: A Rigorous Approach |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.03613) |
|  | Retrieval-augmented Recommender System: Enhancing Recommender Systems with Large Language Models |  |  | RecSys Doctoral Symposium 2023 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3604915.3608889) |
|  | User-Centric Conversational Recommendation: Adapting the Need of User with Large Language Models |  |  | RecSys Doctoral Symposium 2023 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3604915.3608885) |
|  | JobRecoGPT -- Explainable job recommendations using LLMs |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.11805) |
|  | Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2309.10435) |
|  | Lending Interaction Wings to Recommender Systems with Conversational Agents |  |  | NIPS 2023 | [[Link]](https://arxiv.org/abs/2310.04230) |
|  | A Multi-facet Paradigm to Bridge Large Language Model and Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.06491) |
|  | MuseChat: A Conversational Music Recommendation System for Videos |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.06282) |
|  | EcomGPT: Instruction-tuning Large Language Models with Chain-of-Task Tasks for E-commerce |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.06966) |
|  | AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09233) |
|  | Factual and Personalized Recommendations using Language Models and Reinforcement Learning |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.06176) |
|  | A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models |  |  |  Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.09497) |
|  | Language Models As Semantic Indexers |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.07815) |
|  | Multiple Key-value Strategy in Recommendation Systems Incorporating Large Language Model |  |  | CIKM GenRec 2023 | [[Link]](https://arxiv.org/abs/2310.16409) |
|  | LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.17488) |
|  | Improving Conversational Recommendation Systems via Bias Analysis and Language-Model-Enhanced Data Augmentation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.16738) |
|  | Conversational Recommender System and Large Language Model Are Made for Each Other in E-commerce Pre-sales Dialogue |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.14626) |
|  | Enhancing Recommender Systems with Large Language Model Reasoning Graphs |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2308.10835) |
|  | Large Language Model Can Interpret Latent Space of Sequential Recommender |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.20487) |
|  | BTRec: BERT-Based Trajectory Recommendation for Personalized Tours |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.19886) |
|  | Large Multi-modal Encoders for Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2310.20343) |
|  | LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking |  |  | PGAI@CIKM 2023 | [[Link]](https://arxiv.org/abs/2311.02089) |
|  | ITEm: Unsupervised Image-Text Embedding Learning for eCommerce |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.02084) |
|  | Exploring Recommendation Capabilities of GPT-4V(ision): A Preliminary Case Study |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.04199) |
|  | OLaLa: Ontology Matching with Large Language Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.03837) |
|  | Bridging the Information Gap Between Domain-Specific Model and General LLM for Personalized Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.03778) |
|  | Large Language Model based Long-tail Query Rewriting in Taobao Search |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.03758) |
|  | ID Embedding as Subtle Features of Content and Structure for Multimodal Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.05956) |
|  | Exploring Fine-tuning ChatGPT for News Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.05850) |
|  | Do LLMs Implicitly Exhibit User Discrimination in Recommendation? An Empirical Study |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.07054) |
|  | Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.09049) |
|  | RecExplainer: Aligning Large Language Models for Recommendation Model Interpretability |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.10947) |
|  | Knowledge Plugins: Enhancing Large Language Models for Domain-Specific Recommendations |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.10779) |
|  | Adapting LLMs for Efficient, Personalized Information Retrieval: Methods and Implications |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.12287) |
|  | ControlRec: Bridging the Semantic Gap between Language Model and Personalized Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2311.16441) |
|  | LLaRA: Aligning Large Language Models with Sequential Recommenders |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.02445) |
|  | Large Language Models for Intent-Driven Session Recommendations |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.07552) |
|  | Multi-Modality is All You Need for Transferable Recommender Systems |  |  | ICDE 2024 | [[Link]](https://arxiv.org/abs/2312.09602) |
|  | A Unified Framework for Multi-Domain CTR Prediction via Large Language Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10743) |
|  | DRDT: Dynamic Reflection with Divergent Thinking for LLM-based Sequential Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.11336) |
|  | RecPrompt: A Prompt Tuning Framework for News Recommendation Using Large Language Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10463) |
|  | Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.10967) |
|  | The Problem of Coherence in Natural Language Explanations of Recommendations |  |  | ECAI 2023 | [[Link]](https://arxiv.org/abs/2312.11356) |
|  | Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16098) |
|  | Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16159) |
|  | RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16018) |
|  | Large Language Models are Not Stable Recommender Systems |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15746) |
|  | Unlocking the Potential of Large Language Models for Explainable Recommendations |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15661) |
|  | Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15599) |
|  | Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15450) |
|  | The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.15524) |
|  | Empowering Few-Shot Recommender Systems with Large Language Models -- Enhanced Representations |  |  | IEEE Access | [[Link]](https://arxiv.org/abs/2312.13557) |
|  | dIR -- Discrete Information Retrieval: Conversational Search over Unstructured (and Structured) Data with Large Language Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.13264) |
|  | Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via Large Language Models |  |  | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2312.16275) |
|  | Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs |  |  | WSDM 2024 | [[Link]](https://arxiv.org/abs/2312.14345) |
|  | Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.04057) |
|  | ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.03605) |
|  | Combining Embedding-Based and Semantic-Based Models for Post-hoc Explanations in Recommender Systems |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.04474) |
|  | User Embedding Model for Personalized Language Prompting |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.04858) |
|  | Social-LLM: Modeling User Behavior at Scale using Language Models and Social Network Data |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.00893) |
|  | LLMRS: Unlocking Potentials of LLM-Based Recommender Systems for Software Purchase |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.06676) |
|  | LLM-Guided Multi-View Hypergraph Learning for Human-Centric Explainable Recommendation |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.08217) |
|  | Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency |  |  | Arxiv 2024 | [[Link]](https://arxiv.org/abs/2401.10545) |

</p>
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
