# Awesome-LLM-for-RecSys [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of AWESOME papers and resources on the large language model (LLM) related recommender system topics. 

:satisfied: Please check out our survey paper for LLM-enhanced RS: [How Can Recommender Systems Benefit from Large Language Models: A Survey](https://arxiv.org/abs/2306.05817)

To catch up with the latest research progress, this repesitory will be actively maintained as well as our released survey paper. Newly added papers will first appear in ``1.6 Paper Pending List: to be Added to Our Survey Paper`` section.

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
| GENRE | A First Look at LLM-Powered Generative News Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06566) |
| AnyPredict | AnyPredict: Foundation Model for Tabular Prediction | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.12081) |
| LLM4KGC | Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs | PaLM (540B)/ ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.09858v1) |
| TagGPT | TagGPT: Large Language Models are Zero-shot Multimodal Taggers | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03022v1) |
| ICPC | Large Language Models for User Interest Journeys | LaMDA (137B) | Full Finetuning/ Prompt Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.15498) |
| DPLLM | Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models | T5-XL (3B) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.05973) |
| KAR | Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.10933) |
| MINT | Large Language Model Augmented Narrative Driven Recommendations | GPT3 (175B) | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2306.02250) |

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
| TALLRec | TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation | LLaMA (7B) | LoRA | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.00447) |
| PBNR | PBNR: Prompt-based News Recommender System | T5-small (60M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.07862) |
    
<b>1.3.2 Item Generation Task</b>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|
| GPT4Rec | GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation | GPT2 (110M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03879) |
| UP5 | UP5: Unbiased Foundation Model for Fairness-aware Recommendation | T5-base (223M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.12090) |
| VIP5 | VIP5: Towards Multimodal Foundation Models for Recommendation | T5-base (223M) | Layerwise Adater Tuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.14302) |
| P5-ID | How to Index Item IDs for Recommendation Foundation Models | T5-small (61M) | Full Finetuning | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06569) |
| FaiRLLM | Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07609) |
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
| ChatGPT | Uncovering ChatGPT's Capabilities in Recommender Systems | ChatGPT | Frozen | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.02182) |

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
| A Survey on Large Language Models for Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.19860) |
| Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2302.03735) |
| Self-Supervised Learning for Recommender Systems: A Survey | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2203.15876) |

<b>1.5.2 Other Papers</b>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| Recommender Systems with Generative Retrieval | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.05065) |
| Evaluation of Synthetic Datasets for Conversational Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2212.08167v1) |
| Generative Recommendation: Towards Next-generation Recommender Paradigm | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03516) |
    
</p>
</details>

<details><summary><b>1.6 Paper Pending List: to be Added to Our Survey Paper</b></summary>
<p>

| **Name** | **Paper** | **LLM Backbone (Largest)** | **LLM Tuning Strategy** | **Publication** | **Link** |
|:---:|:---|:---:|:---:|:---:|:---:|

</p>
</details>

## 2. Datasets & Benchmarks

The datasets & benchmarks for LLM-related RS topics should maintain the original samantic/textual features, instead of anonymous feature IDs.

<details><summary><b>2.1 Datasets</b></summary>
<p>

| **Dataset** | **RS Scenario** | **Link** |
|:---:|:---:|:---:|
| MovieLens | Movie | [[Link]](https://grouplens.org/datasets/movielens/1m/) |
| Amazon | E-commerce | [[Link]](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) |
| BookCrossing | Book | [[Link]](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) |
| GoodReads | Book | [[Link]](https://mengtingwan.github.io/data/goodreads.html) |
| Anime | Anime | [[Link]](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) |
    
</p>
</details>

<details><summary><b>2.2 Benchmarks</b></summary>
<p>

| **Benchmarks** | **Webcite Link** | **Paper** |
|:---:|:---:|:---:|
| OpenP5 | [[Link]](https://github.com/agiresearch/OpenP5) | [[Paper]](https://arxiv.org/abs/2306.11134) |
| TABLET | [[Link]](https://dylanslacks.website/Tablet) | [[Paper]](https://arxiv.org/abs/2304.13188) |
    
</p>
</details>

## 3. Related Repositories

| **Repo Name** | **Maintainer** |
|:---:|:---:|
| [rs-llm-paper-list](https://github.com/wwliu555/rs-llm-paper-list) | [wwliu555](https://github.com/wwliu555) |
| [awesome-recommend-system-pretraining-papers](https://github.com/archersama/awesome-recommend-system-pretraining-papers) | [archersama](https://github.com/archersama) |
| [LLM4Rec](https://github.com/WLiK/LLM4Rec) | [WLiK](https://github.com/WLiK) |
| [Awesome-LLM4RS-Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers) | [nancheng58](https://github.com/nancheng58) |

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
