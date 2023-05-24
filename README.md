# Awesome-LLM-for-RecSys

Generally two advantages:
1. Open-world knowledge embedded in the pre-trained parameters
    - Textual encoder
    - Reasoning ability
3. Textual features as the unified language to bridge different recommendation domains

## 1. Papers

<details><summary><h3><b>1.1 LLM for Feature Engineering</b></h3></summary>
<p>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| Language Models are Realistic Tabular Data Generators | ICLR 2023 | [[Link]](https://arxiv.org/abs/2210.06280) |
| Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning | ICML 2023 | [[Link]](https://arxiv.org/abs/2211.03044) |
| A First Look at LLM-Powered Generative News Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06566) |

</p>
</details>

<details><summary><h3><b>1.2 LLM as Feature Encoder</b></h3></summary>
<p>

| **Paper** | **Publication** | **Encoded Feature** | **Link** |
|:---|:---:|:---:|:---:|
| U-BERT: Pre-training User Representations for Improved Recommendation | AAAI 2021 | User | [[Link]](https://ojs.aaai.org/index.php/AAAI/article/view/16557) |
|  |  |  |  |
| UNBERT: User-News Matching BERT for News Recommendation | IJCAI 2021 | Item | [[Link]](https://www.ijcai.org/proceedings/2021/462) |
| Pre-trained Language Model based Ranking in Baidu Search | KDD 2021 | Item | [[Link]](https://arxiv.org/abs/2105.11108) |
| Pre-trained Language Model for Web-scale Retrieval in Baidu Search | KDD 2021 | Item | [[Link]](https://arxiv.org/abs/2106.03373) |
| Empowering News Recommendation with Pre-trained Language Models | SIGIR 2021 | Item | [[Link]](https://arxiv.org/abs/2104.07413) |
| Towards Universal Sequence Representation Learning for Recommender Systems | KDD 2022 | Item | [[Link]](https://arxiv.org/abs/2206.05941) |
| Boosting Deep CTR Prediction with a Plug-and-Play Pre-trainer for News Recommendation | COLING 2022 | Item | [[Link]](https://aclanthology.org/2022.coling-1.249/) |
| MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation | SIGIR 2022 | Item | [[Link]](https://dl.acm.org/doi/abs/10.1145/3477495.3531896) |
| Tiny-NewsRec: Effective and Efficient PLM-based News Recommendation | EMNLP 2022 | Item | [[Link]](https://arxiv.org/abs/2112.00944) |
| TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations | Arxiv 2022 | Item | [[Link]](https://arxiv.org/abs/2209.07562) |
| Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders | WWW 2023 | Item | [[Link]](https://arxiv.org/abs/2210.12316) |
|  |  |  |
| CTR-BERT: Cost-effective knowledge distillation for billion-parameter teacher models | ENLSP 2021 | User & Item | [[Link]](https://neurips2021-nlp.github.io/papers/20/CameraReady/camera_ready_final.pdf) |

</p>
</details>

<details><summary><h3><b>1.3 LLM as Core Recommender Model</b></h3></summary>
<p>
    
| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| Language Models as Recommender Systems: Evaluations and Limitations | ICBINB 2021 | [[Link]](https://openreview.net/forum?id=hFx3fY7-m9b) |
| PTM4Tag: Sharpening Tag Recommendation of Stack Overflow Posts with Pre-trained Models | ICPC 2022 | [[Link]](https://arxiv.org/abs/2203.10965) |
| M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2205.08084) |
| PTab: Using the Pre-trained Language Model for Modeling Tabular Data | Arxiv 2022 | [[Link]](https://arxiv.org/abs/2209.08060) |
|  |  |  |
| Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5) | RecSys 2022 | [[Link]](https://arxiv.org/abs/2203.13366) |
| GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03879) |
| Zero-Shot Recommendation as Language Modeling | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2112.04184) |
| How to Index Item IDs for Recommendation Foundation Models | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.06569) |
| Recommender Systems with Generative Retrieval | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.05065) |
| TabLLM: Few-shot Classification of Tabular Data with Large Language Models | AISTATS 2023 | [[Link]](https://arxiv.org/abs/2210.10723) |
| Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction | Arxiv 2023 | [[Link]](https://arxiv.org/pdf/2305.06474.pdf) |
|  |  |  |
| Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07001) |
| TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.00447) |
| PALR: Personalization Aware LLMs for Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07622) |
|  |  |  |
| Zero-Shot Next-Item Recommendation using Large Pretrained Language Models | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03153) |
| Is ChatGPT a Good Recommender? A Preliminary Study | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.10149) |
| Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.09542) |
| Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples! | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2303.08559) |
| Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.04518) |
| Uncovering ChatGPT's Capabilities in Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.02182) |
| Large Language Models are Zero-Shot Rankers for Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.08845) |
| Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07609) |

</p>
</details>

<details><summary><h3><b>1.4 LLM as RS Pipeline Controller</b></h3></summary>
<p>
    
| **Paper** | **Publication** | **Code** |
|:---|:---:|:---:|
| Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2303.14524) |
| Leveraging Large Language Models in Conversational Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2305.07961) |
| Generative Recommendation: Towards Next-generation Recommender Paradigm | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2304.03516) |
</p>
</details>

<details><summary><h3><b>1.5 Related Survey Paper</b></h3></summary>
<p>

| **Paper** | **Publication** | **Link** |
|:---|:---:|:---:|
| Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems | Arxiv 2023 | [[Link]](https://arxiv.org/abs/2302.03735) |

</p>
</details>

## 2. Datasets

| **Dataset** | **# Sample** | **Link** |
|:---|:---:|:---:|
| MovieLens | - | [[Link]](https://grouplens.org/datasets/movielens/1m/) |
| Amazon | - | [[Link]](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) |
| BookCrossing | - | [[Link]](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) |
| GoodReads | - | [[Link]](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) |
| Anime | - | [[Link]](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) |
| TABLET | - | [[Link]](https://dylanslacks.website/Tablet) |
