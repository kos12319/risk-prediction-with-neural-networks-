import pandas as pd

# Create a comprehensive table of research papers found for the thesis
papers_data = {
    'Paper Title': [
        'Credit risk prediction in an imbalanced social lending dataset',
        'Classification based credit risk analysis: The case of Lending Club',
        'Predicting Credit Risk in Peer-to-Peer Lending: A Neural Network Approach',
        'Building a Risk Indicator from Loan Descriptions in P2P Lending (BERT)',
        'Generalist Credit Scoring through Large Language Models',
        'The Evaluation on the Credit Risk of Enterprises (CNN-LSTM-ATT)',
        'A Hybrid CNN-LSTM Model for Enhancing Bond Default Risk Prediction',
        'Credit Risk Modeling with Generative AI and Autonomous Agents',
        'Synthesizing credit data using autoencoders and GANs',
        'Sequential Deep Learning for Credit Risk Monitoring with Tabular Financial Data',
        'Enhancing Credit Risk Assessment Through Transformer-Based Machine Learning Models',
        'Credit Risk Identification in Supply Chains Using GANs',
        'Neural Networks for Credit Risk and xVA in a Front Office Pricing Environment',
        'A new deep learning ensemble credit risk evaluation model',
        'Multi-Modal Deep Learning for Credit Rating Prediction',
        'Credit Risk Assessment Models in Financial Technology',
        'A Complete Revision of LendingClub Predicción del default',
        'Credit risk analysis with machine learning techniques in peer-to-peer lending market',
        'Data Mining Techniques to Predict Default in LendingClub',
        'Multi-view ensemble learning based on distance-to-model for P2P credit risk',
        'A credit risk assessment model of borrowers in P2P lending',
        'Machine learning and artificial neural networks to construct P2P loan credit-scoring model',
        'Kolmogorov–Arnold Networks-based GRU and LSTM for financial risk',
        'FinLangNet: A Novel Deep Learning Framework for Credit Risk',
        'Research on credit risk of listed companies: a hybrid model with Transformer',
        'Credit Risk Prediction of Small and Medium-Sized Enterprises based on LSTM',
        'CNN-based financial risk identification and assessment',
        'Credit Risk BERT: A Pre-Trained Technique for Credit Risk Forecasting',
        'Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy',
        'GPT classifications, with application to credit lending',
        'Explore the Use of Prompt-Based LLM for Credit Risk Assessment'
    ],
    
    'Architecture/Method': [
        'Random Forest + Resampling techniques',
        'Logistic Regression + Random Forest',
        'Feed-forward Neural Network',
        'BERT (Transformer)',
        'LLMs (GPT-3/4, BERT variants)',
        'CNN + LSTM + Attention',
        'CNN + LSTM',
        'GANs + VAEs + LLMs',
        'Autoencoders + GANs',
        'Temporal Convolutional Networks',
        'CNN-SFTransformer + GRU-Transformer',
        'GANs',
        'GRU Neural Networks',
        'Deep learning ensemble',
        'CNN, ConvLSTM, ConvGRU, CNN-Attn, BERT',
        'CNNs + Deep Neural Networks',
        'Random Forest',
        'SVM, Decision Tree, MLP, PNN, Deep Learning',
        'Artificial Neural Networks + Logistic Regression',
        'Multi-view ensemble learning',
        'BP Neural Network with LM algorithm',
        'Machine Learning + ANN methods',
        'KAN-based GRU and LSTM',
        'Deep Learning Framework',
        'Transformer models',
        'LSTM networks',
        'CNN applications',
        'BERT for credit risk',
        'Large Language Models (systematic review)',
        'GPT models for classification',
        'Prompt-based LLMs'
    ],
    
    'Dataset': [
        'Lending Club 2016-2017',
        'Lending Club',
        'Bondora P2P lending',
        'Lending Club',
        'Multiple including Lending Club',
        'Enterprise behavior data',
        'Bond default data',
        'Synthetic data',
        'Credit data',
        'Financial time series',
        'Taiwan, Germany, Australia datasets',
        'Supply chain data',
        'Market and trade data',
        'Imbalanced credit data',
        'Bond, market, financial ratios',
        'Various fintech datasets',
        'Lending Club 2007-2020',
        'P2P lending dataset',
        'Lending Club 2007-2015',
        'P2P lending platforms',
        'Chinese P2P data',
        'P2P loan data',
        'Financial risk data',
        'Credit data',
        'Listed companies data',
        'SME data',
        'Enterprise financial data',
        'German Credit, Lending Club, Kaggle',
        'Financial texts and data',
        'Credit lending applications',
        'Give Me Some Credit dataset'
    ],
    
    'Year': [
        2018, 2022, 2015, 2024, 2023, 2022, 2024, 2020, 2023, 2020, 2024, 2025, 2022, 2021, 2023, 2023, 2023, 2018, 2022, 2020, 2021, 2022, 2025, 2024, 2025, 2022, 2025, 2025, 2025, 2024, 2025
    ],
    
    'Citations_Approx': [
        170, 3, 1, 2, 36, 15, 16, 'N/A', 14, 63, 2, 16, 2, 234, 'N/A', 1, 2, 4, 'N/A', 105, 49, 53, 'N/A', 'N/A', 3, 'N/A', 4, 'N/A', 1, 25, 'N/A'
    ],
    
    'Focus_Area': [
        'Imbalanced learning, P2P lending',
        'Classification, risk analysis',
        'Neural networks for P2P',
        'NLP + Credit risk, BERT',
        'LLMs for credit scoring',
        'CNN-LSTM for enterprises',
        'Hybrid CNN-LSTM for bonds',
        'Generative AI methods',
        'Data synthesis with deep learning',
        'Sequential modeling',
        'Transformer architectures',
        'GANs for credit risk',
        'Front office applications',
        'Ensemble deep learning',
        'Multi-modal approaches',
        'Fintech applications',
        'Feature importance analysis',
        'Comparative ML methods',
        'Traditional vs ANN comparison',
        'Ensemble methods for P2P',
        'BP neural networks',
        'ML and ANN comparison',
        'Novel architectures',
        'Deep learning frameworks',
        'Transformer applications',
        'LSTM for SMEs',
        'CNN applications',
        'Pre-trained models',
        'LLM systematic review',
        'GPT applications',
        'Prompt engineering'
    ],
    
    'Repository_Type': [
        'ArXiv', 'ArXiv', 'IEEE', 'ArXiv', 'ArXiv', 'PMC', 'Journal', 'Course', 'Elsevier', 'ArXiv', 'Conference', 'ArXiv', 'SSRN', 'Elsevier', 'ArXiv', 'Conference', 'Journal', 'Thesis', 'Journal', 'Elsevier', 'PMC', 'Journal', 'ArXiv', 'ArXiv', 'Nature', 'Journal', 'Journal', 'Blog/Tutorial', 'ArXiv', 'Elsevier', 'Journal'
    ]
}

# Create DataFrame
df = pd.DataFrame(papers_data)

# Display the table
print("Comprehensive Research Papers for Credit Risk Prediction with Neural Networks")
print("Focus: Lending Club Dataset and Related P2P/Credit Risk Studies")
print("="*80)
print(df.to_string(index=False))

# Save to CSV for easy access
df.to_csv('credit_risk_neural_networks_research_papers.csv', index=False)
print(f"\n\nTable saved as 'credit_risk_neural_networks_research_papers.csv'")

# Summary statistics
print(f"\n\nSummary Statistics:")
print(f"Total papers found: {len(df)}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
print(f"\nArchitecture/Method distribution:")
arch_counts = df['Focus_Area'].value_counts()
print(arch_counts.head(10))