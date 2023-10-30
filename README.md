## 一、项目简介（Repo’s Brief Introduction）

这里是截至目前关于GPT的一切你应该知道的，包括论文，开源模型，网站，博客... 我将它们整理到一起，以便大家更方便地了解和使用GPT。项目不定期更新，内容可能不全，还请大家补充！

Here is everything about GPT  what you should know so far, including papers, open source models, websites, blogs... I organize them together so that everyone can understand and use GPT more easily. The project is updated from time to time, and the content may be incomplete. Please add it!

## 二、ChatGPT(GPT4)的前世今生（The past and present of ChatGPT (GPT4)）

### 1. LLMs(Large Language Models)(大语言模型)

#### 1.1 GPT1

paper:[Improving Language Understanding by Generative Pretraining](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

code:[openai/finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)

#### 1.2 GPT2

paper:[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

code:[openai/gpt-2](https://github.com/openai/gpt-2)

#### 1.3 GPT3

paper:[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165v4)

code:未开源(Not open source)

#### 1.4 CPM

paper:[CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)

code:[TsinghuaAI/CPM](https://github.com/TsinghuaAI/CPM)

#### 1.5 FastMoE(Foundation of WuDao 2.0)

paper:[FastMoE: A Fast Mixture-of-Expert Training System](https://arxiv.org/abs/2103.13262v1)

code:[laekov/fastmoe](https://github.com/laekov/fastmoe)

#### 1.6 CPM2

paper:[CPM-2: Large-scale Cost-effective Pre-trained Language Models](https://arxiv.org/abs/2106.10715)

code:[TsinghuaAI/CPM](https://github.com/TsinghuaAI/CPM)

#### 1.7 Megatron-LM

paper:[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

code:[NVIDIA/Megatron](https://github.com/NVIDIA/Megatron-LM)

#### 1.8 ERINE 3.0

paper:[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)（base)

code:[PaddleNLP/erine-3.0](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)

#### 1.9 Claude

paper:[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf)

code:未开源(Not open source)

#### 1.10 GLaM

paper:[GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)

code:未开源(Not open source)

#### 1.11 Gopher

papar:[Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf)

code:未开源(Not open source)

#### 1.12 LaMDA

paper:[LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239)

code:未开源(Not open source)

#### 1.13 Turing-NLG-530B

paper:[Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)

code:未开源(Not open source)

#### 1.14 ChinChilla

paper:[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556v1)

code:未开源(Not open source)

#### 1.15 GPT3.5(InstructionGPT)

paper(有两篇供参考)(There are two ones to be referred):

[Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)

[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)																		  

code:未开源(Not open source)

#### 1.16 PaLM

paper:[PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)

code:[lucidrains/PaLM-pytorch](https://github.com/lucidrains/PaLM-pytorch)

#### 1.17 OPT

paper:[OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)

code:[facebookresearch/metaseq](https://github.com/facebookresearch/metaseq)

#### 1.18 BaGuaLu

paper:[BaGuaLu: Targeting Brain Scale Pretrained Models with over 37 Million Cores](https://dl.acm.org/doi/pdf/10.1145/3503221.3508417)

code:未开源(Not open source)

#### 1.19 Minerva

paper:[Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)

code:未开源(Not open source)

#### 1.20 BLOOM

paper:[BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)

code:

[huggingface/bigscience/bloom](https://huggingface.co/bigscience/bloom)

[bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)

#### 1.21 GLM
paper:[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360)

code:[THUDM/GLM](https://github.com/THUDM/GLM)

#### 1.22 GLM-130B

paper:[GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL](https://arxiv.org/pdf/2210.02414.pdf)

code:[THUDM/GLM-130B](https://github.com/THUDM/GLM-130B)

#### 1.23 LLaMA

paper:[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

code:[facebookresearch/llama](https://github.com/facebookresearch/llama)

#### 1.24 LLAMA2

paper:[LLAMA2:Open Foundation and Fine-Tuned Chat Models](https://scontent-lax3-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=ByL78P2ckIMAX8z5u_N&_nc_ht=scontent-lax3-1.xx&oh=00_AfAVpYZQDzxoFVdlKL7ShJuiI372Us5Buygs31qZXgUzUQ&oe=64C84A3F)

code:[facebookresearch/llama](https://github.com/facebookresearch/llama)

#### 1.25 Alpaca

code:[tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

#### 1.26 GPT4

paper:[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

code:未开源(Not open source)

#### 1.27 Vicuna/FastChat

code:[lm-sys/FastChat](https://github.com/lm-sys/FastChat)

#### 1.28 Cerabras-GPT

paper:[Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster](https://arxiv.org/abs/2304.03208)

code:[huggingface/cerabras](https://huggingface.co/cerebras)

#### 1.29 PanGu-α

paper:[PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation](https://arxiv.org/abs/2104.12369v1)

code:[huawei-noah/Pretrained-Language-Model/PanGu-α](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-α)

#### 1.30 PanGu-Σ

paper:[PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing](https://arxiv.org/abs/2303.10845)

code:未开源(Not open source)

#### 1.31 Yuan 1.0

paper:[Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning](https://arxiv.org/abs/2110.04725)

code:未开源(Not open source)

#### 1.32 Mengzi

paper:[Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese](https://arxiv.org/abs/2110.06696)

code:[Langboat/Mengzi](https://github.com/Langboat/Mengzi)

#### 1.33 PaLM2

paper:[PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf)

code:未开源(Not open source)

#### 1.34 LLaMA-Adapter

paper:[LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/abs/2303.16199)

code:[ZrrSkywalker/LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter)

#### 1.35 CPM-Bee

code:[OpenBMB/CPM-Bee](https://github.com/OpenBMB/CPM-Bee)

#### 1.36 Baichuan2

code:[baichuan-inc/Baichuan-7B](https://github.com/baichuan-inc/Baichuan2)

### 2.Embeedings

#### 2.1 Word2Vec

paper:[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

code:[Tensorflow Core/Word2Vec](https://www.tensorflow.org/tutorials/text/word2vec)

#### 2.2 Doc2Vec

paper:[Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)

code:[Gensim/Doc2Vec Model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)

(Gensim is a popular python nlp processing library)

#### 2.3 Context2Vec

paper:[context2vec: Learning Generic Context Embedding with Bidirectional LSTM](https://aclanthology.org/K16-1006.pdf)

code:[orenmel/context2Vec](https://github.com/orenmel/context2vec)

#### 2.4 lda2Vec

paper:[Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019v1)

code:[cemoody/lda2vec](https://github.com/cemoody/lda2vec)

#### 2.5 TWEC

paper:[Training Temporal Word Embeddings with a Compass](https://arxiv.org/abs/1906.02376v1)

code:[valedica/twec](https://github.com/valedica/twec)

#### 2.6 USE

paper:[Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/abs/1907.04307v1)

code:[Dimitre/universal-sentence-encoder](https://huggingface.co/Dimitre/universal-sentence-encoder)

#### 2.7 fastText

paper:[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606v2)

code:[facebookresearch/fastText](https://github.com/facebookresearch/fastText)

#### 2.8 ELMo

paper:[Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)

code:[HIT-SCIR/ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs)

#### 2.9 GloVe

paper:[GloVe:Global Vectors for Word Representation](https://www-nlp.stanford.edu/pubs/glove.pdf)

code:[stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe)

### 3.Transformer and Its Variants

#### 3.1 Transformer

paper:[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

code:[tunz/transformer-pytorch](https://github.com/tunz/transformer-pytorch)

#### 3.2 BERT

paper:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805v2)

code:[google-research/bert](https://github.com/google-research/bert)

#### 3.3 T5

paper:[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683v3)

code:[google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)

#### 3.4 Reformer

paper:[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451v2)

code:[google/trax/reformer](https://github.com/google/trax/tree/master/trax/models/reformer)

#### 3.5 Longformer

paper:[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150v2)

code:[allenai/longformer](https://github.com/allenai/longformer)

#### 3.6 XLM

paper:[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291v1)

code:[facebookresearch/XLM](https://github.com/facebookresearch/XLM)

#### 3.7 UniLM

paper:[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)

code:[microsoft/unilm](https://github.com/microsoft/unilm)

#### 3.8 RoBERTa

paper:[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692v1)

code:[huggingface/roberta](https://huggingface.co/docs/transformers/model_doc/roberta)

#### 3.9 ALBERT

paper:[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

code:[google-research/albert](https://github.com/google-research/albert)

#### 3.10 DeBERTa

paper:[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654v6)

code:[microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)

#### 3.11 BART

paper:[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461v1)

code:[huggingface/bart](https://huggingface.co/docs/transformers/model_doc/bart)

#### 3.12 XLNet

paper:[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237v2)

code:[zihangdai/xlnet](https://github.com/zihangdai/xlnet)

#### 3.13 ViT

paper:[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

code:[google-research/vision_transformer](https://github.com/google-research/vision_transformer)

#### 3.14 Swin Transformer

paper:[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

code:[microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

#### 3.15 DistillBERT

paper:[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108v4)

code:[huggingface/distilbert-base-uncased-distilled-squad](https://huggingface.co/distilbert-base-uncased-distilled-squad)

#### 3.16 Switch Transformer

paper:[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

code:[LoniQin/english-spanish-translation-switch-transformer](https://github.com/LoniQin/english-spanish-translation-switch-transformer) (Not Source Code,just an example)

#### 3.17 Mirror-BERT

paper:[Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders](https://arxiv.org/abs/2104.08027v2)

code:[cambridgeltl/mirror-bert](https://github.com/cambridgeltl/mirror-bert)

#### 3.18 Charformer

paper:[Charformer: Fast Character Transformers via Gradient-based Subword Tokenization](https://arxiv.org/abs/2106.12672v3)

code:[google-research/charformer](https://github.com/google-research/google-research/tree/master/charformer)

#### 3.19 Big Bird

paper:[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)

code:[google-research/bigbird](https://github.com/google-research/bigbird)

#### 3.20 ELECTRA

paper:[ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS](https://arxiv.org/pdf/2003.10555.pdf)

code:[google-research/electra](https://github.com/google-research/electra)

#### 3.21 Gshard

paper:[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/pdf/2006.16668.pdf)

### 4. Attention

#### 4.1 Self-Attention 

paper:[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

code:[tunz/transformer-pytorch](https://github.com/tunz/transformer-pytorch)

#### 4.2 Multi-head Attention

paper:[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

code:[tunz/transformer-pytorch](https://github.com/tunz/transformer-pytorch)

#### 4.3 Addictive Attention

paper:[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7)

code:[Tensorflow/AddictiveAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AdditiveAttention)

#### 4.4 Global-Local Attention

paper:[ETC: Encoding Long and Structured Inputs in Transformers](https://arxiv.org/abs/2004.08483v5)

code:[google-research/etcmodel](https://github.com/google-research/google-research/tree/master/etcmodel)

#### 4.5 Sparse Attention

paper:[Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509v1.pdf)

code:[openai/sparse_attention](https://github.com/openai/sparse_attention)

#### 4.6 Routing Attention

paper:[Efficient Content-Based Sparse Attention with Routing Transformers](https://arxiv.org/abs/2003.05997v5)

code:[lucidrains/routing-transformer](https://github.com/lucidrains/routing-transformer/)

### 5. Others

#### 5.1 n-gram

paper:[N-gram Language Model](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (Not Original Paper)

#### 5.2 skip-gram

paper:[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

code:[SeanLee97/nlp_learning/word2vec/cbow](https://github.com/SeanLee97/nlp_learning/tree/master/word2vec/cbow)

#### 5.3 Bag of Words(BOW)

paper:[An Overview of Bag of Words;Importance, Implementation, Applications, and Challenges](https://www.researchgate.net/publication/338511771_An_Overview_of_Bag_of_WordsImportance_Implementat) (Not Original Paper)

#### 5.4 CBOW

paper:[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

code:[SeanLee97/nlp_learning/word2vec/skipgram](https://github.com/SeanLee97/nlp_learning/tree/master/word2vec/skipgram)

#### 5.5 Seq2Seq

paper:[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215v3)

code:[farizrahman4u/seq2seq](https://github.com/farizrahman4u/seq2seq)

#### 5.6 WordPiece

paper:[Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144v2)

code:[google/sentencepiece](https://github.com/google/sentencepiece)

#### 5.7 SentencePiece

paper:[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226v1)

code:[google/sentencepiece](https://github.com/google/sentencepiece)

#### 5.8 RNN

paper:[Find Structure in Time](http://psych.colorado.edu/~kimlab/Elman1990.pdf)

code:[tensorflow/keras/RNN](https://www.tensorflow.org/guide/keras/rnn) (Not Source Code)

#### 5.9 LSTM

paper:[LONG SHORT-TERM MEMORY](https://www.bioinf.jku.at/publications/older/2604.pdf)

code:[tensorflow/keras/LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) (Not Source Code)

#### 5.10 GRU

paper:[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078v3)

code:[tensorflow/keras/GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU) (Not Source Code)

#### 5.11 BiLSTM

paper:[Framewise Phoneme Classification with Bidirectional LSTM Networks](https://www.cs.toronto.edu/~graves/ijcnn_2005.pdf)

code:[tensorflow/keras/bidirectional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional) (Not Source Code)

## 三、已有的类ChatGPT产品及基于ChatGPT的应用(Existing ChatGPT-like products and applications based on ChatGPT)

### 1 BriefGPT

intro:Arxiv论文速递

web:https://briefgpt.xyz/

### 2 一起用AI、GPT3demo、Futurepedia、AI Library、AI工具集

intro:collections of AI products(tools) and gpt-based products(tools)

web:

https://17yongai.com

https://gpt3demo.com/

https://www.futurepedia.io/

https://library.phygital.plus/

https://ai-bot.cn/

### 3 灵羽助手

web:https://www.ai-anywhere.com/(目前只有电脑客户端)

### 4 Catalyst Plus(催化剂+)

web:https://www.researchercosmos.com/(目前大部分功能只为客户端提供)

### 5 Notion AI

intro:Write notes intelligently

web:https://www.notion.so/ (inside Notion)

### 6 Microsoft 365 AI

Inside Microsoft 365 (Not available in some countries)

### 7 WPS AI

Inside WPS(A Chinese Microsoft 365-like Product)

### 8 Windows Copilot

Inside Windows (Not available in some countries)

### 9 飞书"My AI"

Inside 飞书(Lark) (A Chinese office software) (Not currently open to the public)

### 10 钉钉"/"(魔法棒)

Inside 钉钉 (A Chinese office software) (Not currently open to the public)

### 11 文心一言 (百度 Baidu)

web:https://yiyan.baidu.com/welcome

### 12 通义千问 (阿里巴巴 Alibaba)

web:https://tongyi.aliyun.com/

### 13 商量 (商汤科技 SenseTime)

web:https://chat.sensetime.com/

### 14 讯飞星火认知大模型 (科大讯飞 iFLYTEK)

web:https://xinghuo.xfyun.cn/ 

### 15 MOSS (复旦大学 FDU)

code:[OpenLMlab/MOSS](https://github.com/OpenLMLab/MOSS)

web:https://moss.fastnlp.top/ (Not currently open to the public)

### 16 曹植 (达观数据 Data Grand)

web:http://www.datagrand.com/products/aigc/ (Not currently open to the public)

### 17 天工AI助手 (昆仑万维 & 奇点智源)

web:https://tiangong.kunlun.com/

### 18 奇妙文(序列猴子) (出门问问 Mobvoi)

web:https://wen.mobvoi.com/

### 19 式说(SageGPT) (第四范式 4Paradigm)

web:http://www.4paradigm.com/product/SageGPT.html (Not currently open to the public)

### 20 从容(云从科技 CloudWalk)

web:https://maas.cloudwalk.com/ (Not currently open to the public)

### 21 面壁露卡(Luca) (面壁智能 Modelbest)

web:https://luca-beta.modelbest.cn/

### 22 360智脑

web:https://ai.360.cn/

### 23 TigerBot

code:[TigerResearch/TigerBot](https://github.com/TigerResearch/TigerBot)

web:https://www.tigerbot.com/

### 24 山海(云知声 UniSound)

web:https://shanhai.unisound.com/

### 25 智谱青言(Based on ChatGLM2 developed by THUDM)

code:https://github.com/THUDM/ChatGLM2-6B

web:https://chatglm.cn/

### 26 百川大模型(百川智能)

web:https://chat.baichuan-ai.com/chat

### 27 豆包(字节跳动 ByteDance)

web:https://www.doubao.com/chat/

### 28 MathGPT(好未来)

web:www.mathgpt.com

### 29 Bard

web:https://bard.google.com/

### 30 Claude2(The Newest Version of Claude developed by Anthropic)

intro:The Strongest Competitor of ChatGPT

web:https://claude.ai/

### 31 AutoGPT

code:[Significant-Gravitas/Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT)

### 32 AgentGPT (AutoGPT Powered)

web:https://agentgpt.reworkd.ai/

code:[reworkd/AgentGPT](https://github.com/reworkd/AgentGPT)

### 33 MiniGPT-4

code:[Vision-CAIR](https://github.com/Vision-CAIR/MiniGPT-4)

### 34 HuggingChat

web:https://huggingface.co/chat

### 35 Perplexity AI

web:https://www.perplexity.ai/

### 36 Chat with Open Large Language Models

web:https://chat.lmsys.org/

### 37 FactGPT

web:https://factgpt-fe.vercel.app/

### 38 GPTZero

web:https://gptzero.me/

## 四、GPT和LLM的理论研究与应用研究(Theoretical and applied research of GPT and LLM)

## 五、关于GPT和LLM的博客与文章（主要来自微信公众号和Medium）（Blogs and articles about GPT and LLM (mainly from WeChat public account and Medium)）

## 六、写在最后(Write At The End)

人工智能（AI）的发展日新月异，AI的进步速度远远超乎我们的想象，我们应该始终保持学习的动力，积极主动拥抱AI新时代。但同时，也要看到大规模应用AI所带来的潜在风险和LLM的局限性。因此，我们应该拥有独立思考的能力，辩证看待AI，AIGC和LLM的发展。无论如何，我们的终极目标都是让AI造福人类，造福世界。

The development of Artificial Intelligence(AI) is changing with each passing day, and the speed of AI progress is far beyond our imagination. We should always maintain the motivation of learning and actively embrace the new era of AI. But at the same time, we must also see the potential risks brought about by the large-scale application of AI and the limitations of LLM. Therefore, we should have the ability to think independently and look at the development of AI, AIGC and LLM dialectically. In any case, our ultimate goal is to make AI benefit mankind and the world.
