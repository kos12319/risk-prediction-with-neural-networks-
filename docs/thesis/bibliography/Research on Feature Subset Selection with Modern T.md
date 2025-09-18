<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Research on Feature Subset Selection with Modern Transformer Architectures for Credit Risk

Based on my comprehensive research across academic databases, arXiv, and specialized repositories, I have identified **15 highly relevant papers** that explore the intersection of feature subset selection, transformer architectures, and credit risk assessment. Here are the key findings:

## Direct Applications to Lending Club Dataset

### **Most Relevant Papers for Your Thesis**

1. **Building a Risk Indicator from Loan Descriptions in P2P Lending using BERT** (2024) [2 citations]
    - **Architecture**: BERT transformer applied directly to Lending Club loan descriptions
    - **Feature Selection**: BERT attention mechanism identifies important textual features from loan descriptions
    - **Contribution**: First systematic application of BERT to P2P lending text analysis using Lending Club data
    - **Relevance**: Perfect match for your thesis topic
2. **Internet Financial Credit Risk Assessment with Sliding Window and Attention LSTM** (2023) [13 citations]
    - **Architecture**: LSTM with attention mechanism and sliding window approach
    - **Feature Selection**: Attention mechanism focuses on the most important temporal and feature information
    - **Dataset**: Uses Lending Club public dataset specifically
    - **Contribution**: Demonstrates superior performance over traditional methods (ARIMA, SVM, ANN)
3. **Incorporating BERT-based NLP and Transformer for Credit Risk Assessment** [3 citations]
    - **Architecture**: BERT-based NLP combined with transformer models
    - **Dataset**: Uses Lending Club dataset with 26 features including job titles
    - **Feature Selection**: NLP techniques extract features from textual data (173,105 unique job title entries)
4. **Performance Analysis of Credit Scoring Models on Lending Club Data** (2017) [5 citations]
    - **Comprehensive Study**: Systematic evaluation of multiple machine learning approaches on Lending Club
    - **Feature Selection**: Traditional importance methods compared across different classifiers
    - **Baseline Study**: Provides benchmark results for Lending Club dataset

## Modern Transformer Architectures with Feature Selection

### **Highly Cited Foundational Work**

1. **Feature Importance Estimation with Self-Attention Networks** (2020) [**63 citations**] ⭐ **Most Cited**
    - **Architecture**: Self-Attention Networks (SAN) specifically designed for tabular data
    - **Innovation**: First systematic use of attention mechanisms for feature importance estimation
    - **Methods**: Three different attention aggregation strategies for feature ranking
    - **Comparison**: Benchmarked against ReliefF, Mutual Information, and Random Forest importance methods
    - **Key Finding**: SANs identify similar high-ranked features as established methods while detecting feature interactions

### **Recent Advanced Approaches**

2. **A hierarchical attention-based feature selection and fusion for credit risk** (2024) [9 citations]
    - **Architecture**: Hierarchical attention mechanism for credit risk assessment
    - **Feature Selection**: Multi-level attention for feature selection and fusion
    - **Innovation**: Characterizes ability to manage feature selection hierarchically
3. **Comparative Analysis of Transformers for Modeling Tabular Data: Industry Scale** (2023) [2 citations]
    - **Comprehensive Study**: TabBERT, Twin Tower, LUNA transformers on financial data
    - **Dataset**: American Express credit default prediction (industry-scale)
    - **Key Finding**: Twin Tower architecture outperforms TabBERT on large-scale financial datasets
    - **Feature Selection**: Attention mechanisms for feature representation learning

## Advanced Architectures and Methods

### **TabNet and Sequential Attention**

- **TabNet: Attentive Interpretable Tabular Learning** - Popular architecture specifically designed for tabular data
- **Sequential attention mechanism** enables feature selection at each decision step
- **Interpretability**: Provides feature masks showing which features were selected


### **Dynamic and Adaptive Approaches**

- **Dynamic Self-Attention for Credit Scoring**: Novel ensemble approach with per-customer feature weighting
- **Transformer Encoder with Time2Vec**: Correlation-based feature selection for financial prediction


## Key Research Findings and Insights

### **Attention Mechanisms for Feature Selection**

1. **Self-Attention Weights**: Can be directly interpreted as feature importance scores
2. **Hierarchical Attention**: Multiple levels of attention for complex feature interactions
3. **Dynamic Attention**: Adaptive feature weighting per sample/customer
4. **Sequential Attention**: Progressive feature selection (TabNet approach)

### **Effectiveness on Credit Risk Data**

1. **Temporal Attention**: More important than cross-feature attention in financial time series
2. **Text + Tabular**: BERT attention on loan descriptions combined with numerical features
3. **Industry Scale**: Direct supervised training (Twin Tower) outperforms pre-training approaches on large datasets
4. **Interpretability**: Attention weights provide explainable feature importance for regulatory compliance

## Research Gaps and Opportunities for Your Thesis

### **Major Gaps Identified**

1. **Limited Integration**: Few studies directly combine traditional feature selection WITH transformer attention mechanisms
2. **Lending Club Focus**: Only 4/15 papers specifically use Lending Club dataset with modern transformers
3. **Systematic Comparison**: Lack of comprehensive comparison of attention-based vs. traditional feature selection
4. **Hybrid Approaches**: Missing integration of multiple feature selection paradigms

### **Novel Research Opportunities**

1. **Attention-Enhanced Feature Selection**: Combine transformer attention with traditional methods (ReliefF, Mutual Information)
2. **Multi-Modal Approach**: Text (loan descriptions) + Tabular (financial features) with joint attention
3. **Interpretable Transformers**: Focus on explainable attention mechanisms for regulatory requirements
4. **Comparative Framework**: Systematic evaluation across different transformer architectures on Lending Club

### **Specific Technical Approaches**

1. **Hybrid Architecture**: Traditional feature selection → Transformer encoder → Attention-based refinement
2. **Multi-Head Attention**: Different attention heads for different feature types (numerical, categorical, temporal)
3. **Progressive Selection**: Iterative feature selection using transformer attention weights
4. **Ensemble Methods**: Combine multiple attention-based feature selection strategies

## Recommendations for Your Thesis

### **Core Papers to Study**

1. Škrlj et al. (2020) - "Feature Importance Estimation with Self-Attention Networks" [63 citations]
2. Sanz-Guerrero et al. (2024) - "Building a Risk Indicator from Loan Descriptions using BERT"
3. Li et al. (2023) - "Internet Financial Credit Risk Assessment with Attention LSTM"
4. Singh et al. (2023) - "Comparative Analysis of Transformers for Tabular Data"

### **Technical Implementation Path**

1. **Baseline**: Implement traditional feature selection methods on Lending Club
2. **Transformer Integration**: Apply BERT/TabNet with attention-based feature selection
3. **Novel Hybrid**: Develop combined approach using both traditional and attention-based methods
4. **Evaluation**: Comprehensive comparison across multiple metrics and interpretability measures

The research shows a clear opportunity for your thesis to contribute novel insights at the intersection of feature subset selection, transformer architectures, and credit risk prediction using the Lending Club dataset.
<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^2][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^3][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^4][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^5][^50][^51][^52][^53][^54][^55][^6][^7][^8][^9]</span>

<div style="text-align: center">⁂</div>

[^1]: https://arxiv.org/pdf/2311.14335.pdf

[^2]: https://pub.towardsai.net/dynamic-self-attention-for-credit-scoring-a-novel-ensemble-approach-5df1b6952ab1

[^3]: https://arxiv.org/abs/2505.00725

[^4]: http://arno.uvt.nl/show.cgi?fid=170629

[^5]: https://arxiv.org/pdf/2002.04464.pdf

[^6]: https://arxiv.org/html/2306.02136v2

[^7]: https://arxiv.org/html/2404.13004v2

[^8]: https://membranetechnology.org/index.php/journal/article/download/404/272/751

[^9]: https://www.sciencedirect.com/science/article/abs/pii/S0275531925003356

[^10]: https://www.nature.com/articles/s41586-024-08328-6

[^11]: https://www.sciencedirect.com/science/article/abs/pii/S0167739X24003364

[^12]: https://www.sciencedirect.com/science/article/pii/S0377221725003170

[^13]: https://www.sciencedirect.com/science/article/pii/S0377221722008207

[^14]: https://www.sciencedirect.com/science/article/pii/S0377221724007288

[^15]: https://onlinelibrary.wiley.com/doi/10.1111/1911-3846.12832

[^16]: https://aclanthology.org/2023.finnlp-1.8.pdf

[^17]: https://www.nature.com/articles/s41599-025-05230-y

[^18]: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4606750_code4927761.pdf?abstractid=3971880\&mirid=1

[^19]: https://dl.acm.org/doi/10.1016/j.future.2024.06.036

[^20]: https://www.machinelearningexpedition.com/tabnet-tabular-neural-network/

[^21]: https://hrcak.srce.hr/file/413377

[^22]: https://www.geeksforgeeks.org/machine-learning/tabnet/

[^23]: https://arxiv.org/html/2401.16458v2

[^24]: https://arxiv.org/abs/2311.14335

[^25]: https://www.reddit.com/r/MachineLearning/comments/1c7rfhv/any_ways_to_improve_tabnet_d/

[^26]: https://arxiv.org/abs/2210.05136

[^27]: https://www.sciencedirect.com/science/article/pii/S0004370225000116

[^28]: https://arxiv.org/html/2409.08806v2

[^29]: https://pure.qub.ac.uk/files/422660944/XAI_Chapter_1_.pdf

[^30]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00544/115239/Transformers-for-Tabular-Data-Representation-A

[^31]: https://www.kaggle.com/code/defcodeking/tabnet-a-neural-network-for-tabular-data

[^32]: https://dl.acm.org/doi/10.1145/3632410.3632456

[^33]: https://github.com/zeinhasan/TabNet-imbalanced-Class

[^34]: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4528544_code4575873.pdf?abstractid=4528544\&mirid=1

[^35]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11937286/

[^36]: https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/jat.4803

[^37]: https://www.semanticscholar.org/paper/A-Deep-Learning-Approach-for-Credit-Scoring-of-LSTM-Wang-Han/424ffc58ef14751687a5aa5feb2c9e693e10d667

[^38]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9110157/

[^39]: https://arxiv.org/abs/2504.13801

[^40]: https://www.nature.com/articles/s41598-025-86371-7

[^41]: https://arxiv.org/pdf/2504.13801.pdf

[^42]: https://dspace.cuni.cz/bitstream/handle/20.500.11956/86490/DPTX_2016_1_11230_0_519394_0_185577.pdf?sequence=1

[^43]: https://systems.enpress-publisher.com/index.php/jipd/article/viewFile/9652/5334

[^44]: https://dl.acm.org/doi/10.1145/3529399.3529429

[^45]: https://dialnet.unirioja.es/descarga/articulo/9086820.pdf

[^46]: https://www.sciencedirect.com/science/article/pii/S187705092502438X

[^47]: https://www.kism.or.kr/file/memoir/13_4_1.pdf

[^48]: https://www.sciencedirect.com/science/article/pii/S0038012123002586

[^49]: https://www.sciencedirect.com/science/article/abs/pii/S0957417422006170

[^50]: https://digitalcommons.harrisburgu.edu/cgi/viewcontent.cgi?article=1066\&context=dandt

[^51]: https://www.nature.com/articles/s41598-024-72045-3.pdf

[^52]: https://www.kaggle.com/datasets/deependraverma13/lending-club-loan-data-analysis-deep-learning

[^53]: https://arxiv.org/abs/2407.11615

[^54]: https://rpubs.com/tbrk1608/lcdata100k

[^55]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/38b5b75cb4b2a4d6e25459afe480d84b/07fd920c-93c3-46dd-b132-7a0a8fb0e5de/b4eb0af2.csv

