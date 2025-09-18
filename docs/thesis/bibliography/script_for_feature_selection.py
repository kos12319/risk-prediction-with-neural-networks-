# Create a comprehensive analysis of the research findings
feature_selection_papers = {
    'Paper Title': [
        'Feature Importance Estimation with Self-Attention Networks',
        'Comparative Analysis of Transformers for Modeling Tabular Data: Industry Scale',
        'A hierarchical attention-based feature selection and fusion for credit risk',
        'Building a Risk Indicator from Loan Descriptions in P2P Lending using BERT',
        'Internet Financial Credit Risk Assessment with Sliding Window and Attention LSTM',
        'TabNet: Attentive Interpretable Tabular Learning',
        'A Transformer-based Model Integrated with Feature Selection',
        'Transformer Encoder and Multi-features Time2Vec for Financial Prediction',
        'A Path-Based Feature Selection Algorithm for Enterprise Credit Risk',
        'Dynamic Self-Attention for Credit Scoring: A Novel Ensemble Approach',
        'Performance Analysis of Credit Scoring Models on Lending Club Data',
        'FinBERT-QA: Financial Question Answering with pre-trained BERT',
        'Financial sentiment analysis using FinBERT with LSTM',
        'Incorporating BERT-based NLP and Transformer for Credit Risk Assessment',
        'RNN Under the Lens: Attention, Confidence, and Feature Importance'
    ],
    
    'Architecture/Method': [
        'Self-Attention Networks (SAN) for feature importance',
        'TabBERT, Twin Tower, LUNA transformers',
        'Hierarchical attention mechanism',
        'BERT for P2P lending risk',
        'LSTM with attention and sliding window',
        'TabNet with sequential attention',
        'Transformer with feature selection',
        'Transformer Encoder with Time2Vec',
        'Path-based feature selection',
        'Dynamic self-attention ensemble',
        'Multiple ML models on Lending Club',
        'FinBERT for financial QA',
        'FinBERT + LSTM for sentiment analysis',
        'BERT-based NLP for Lending Club',
        'RNN with attention and feature importance'
    ],
    
    'Feature Selection Method': [
        'Attention weights as feature importance scores',
        'Attention mechanism for feature representation',
        'Hierarchical attention for feature fusion',
        'BERT attention on loan descriptions',
        'Attention mechanism to focus on important features',
        'Sequential attention for feature selection',
        'Integrated feature selection with transformer',
        'Correlation feature selection method',
        'Path-based feature ranking',
        'Dynamic attention for per-customer feature weighting',
        'Traditional feature importance methods',
        'BERT embeddings for feature extraction',
        'Sentiment-based feature engineering',
        'NLP feature extraction from text',
        'Gradient-based feature importance'
    ],
    
    'Lending Club Usage': [
        'No - general tabular data',
        'No - American Express credit data',
        'No - general credit risk',
        'Yes - Lending Club loan descriptions',
        'Yes - Lending Club dataset',
        'No - general tabular data',
        'No - general applications',
        'No - stock market data',
        'No - enterprise data',
        'No - synthetic credit data',
        'Yes - Lending Club performance analysis',
        'No - financial QA',
        'No - stock market prediction',
        'Yes - Lending Club features',
        'No - IMDB sentiment data'
    ],
    
    'Citations': [63, 2, 9, 2, 13, 'Popular tool', 'Unknown', 1, 5, 'Blog/Tutorial', 5, 1, 'Recent', 3, 'Recent'],
    
    'Key Contribution': [
        'First attention-based feature importance for tabular data',
        'Comprehensive transformer comparison on financial data',
        'Hierarchical attention for feature selection and fusion',
        'BERT application to P2P lending text analysis',
        'Attention LSTM with sliding window for credit risk',
        'Sequential attention mechanism for interpretable tabular learning',
        'Integrated approach combining transformers and feature selection',
        'Novel correlation feature selection for financial prediction',
        'Path-based features for SME credit risk evaluation',
        'Dynamic per-sample feature weighting with attention',
        'Comprehensive evaluation of ML methods on Lending Club',
        'FinBERT adaptation for financial question answering',
        'Sentiment analysis enhancement for financial prediction',
        'NLP techniques applied to Lending Club dataset',
        'Multi-component RNN enhancement with attention'
    ],
    
    'Year': [2020, 2023, 2024, 2024, 2023, 2019, 'Recent', 2025, 2022, 2025, 2017, 2025, 2023, 'Recent', 2025],
    
    'Relevance_Score': [9, 8, 9, 10, 9, 8, 7, 6, 7, 8, 8, 6, 7, 9, 7]
}

import pandas as pd
df_features = pd.DataFrame(feature_selection_papers)

print("Feature Subset Selection with Modern Transformer Architectures")
print("Focus: Credit Risk and Tabular Data Applications")
print("="*70)
print(df_features.to_string(index=False))

# High relevance papers for the thesis topic
high_relevance = df_features[df_features['Relevance_Score'] >= 9].copy()

print("\n\nHighest Relevance Papers for Thesis Topic")
print("(Feature Subset Selection + Transformers + Credit Risk)")
print("="*60)
print(high_relevance[['Paper Title', 'Architecture/Method', 'Lending Club Usage', 'Citations', 'Key Contribution']].to_string(index=False))

# Save comprehensive analysis
df_features.to_csv('feature_selection_transformer_credit_risk_papers.csv', index=False)
print(f"\n\nTable saved as 'feature_selection_transformer_credit_risk_papers.csv'")

# Analysis summary
print(f"\n\nAnalysis Summary:")
print(f"Total papers analyzed: {len(df_features)}")
print(f"Papers using Lending Club dataset: {len(df_features[df_features['Lending Club Usage'].str.contains('Yes')])}")
print(f"High relevance papers (score ≥9): {len(high_relevance)}")
print(f"Papers with >50 citations: {len([c for c in df_features['Citations'] if isinstance(c, int) and c > 50])}")

# Key gaps and opportunities
print(f"\n\nKey Research Gaps Identified:")
print("1. Limited direct combination of feature selection WITH transformer architectures")
print("2. Few studies specifically on Lending Club dataset with modern transformers")
print("3. Lack of systematic comparison of attention-based feature selection methods")
print("4. Missing integration of traditional feature selection with transformer attention")
print("5. Insufficient focus on interpretable feature subset selection in credit risk")