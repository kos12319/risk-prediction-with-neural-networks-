# Create a more detailed categorization by neural network architecture
architecture_summary = {
    'Category': [
        'Traditional Neural Networks',
        'Convolutional Neural Networks (CNNs)',
        'Recurrent Networks (LSTM/GRU)',
        'Hybrid CNN-LSTM/GRU',
        'Transformer-based (BERT/GPT)',
        'Generative Models (GANs/VAEs)',
        'Ensemble Methods',
        'Novel Architectures',
        'Large Language Models'
    ],
    
    'Papers_Count': [5, 4, 6, 8, 7, 3, 4, 3, 6],
    
    'Key_Papers': [
        'Feed-forward NN (Byanjankar 2015), BP Neural Network (Ma 2021)',
        'CNN enterprise risk (Zhang 2022), CNN financial risk (Wang 2025)', 
        'LSTM SMEs (Credit Risk Prediction 2022), GRU xVA (Frodé 2022)',
        'CNN-LSTM-ATT (Zhang 2022), CNN-LSTM bonds (Yao 2024), ConvLSTM (Multi-modal 2023)',
        'BERT P2P (Sanz-Guerrero 2024), GPT credit (Babaei 2024), Transformer risk (Shen 2025)',
        'GANs synthesis (Oreski 2023), GANs supply chain (Zhang 2025), VAEs course (FermaC 2020)',
        'Deep ensemble (Shen 2021), Multi-view ensemble (Song 2020), RF ensemble (Namvar 2018)',
        'Kolmogorov-Arnold Networks (KAN-GRU/LSTM 2025), FinLangNet (2024)',
        'GPT-3/4 scoring (Feng 2023), LLM systematic review (Golec 2025), Prompt-based LLM (2025)'
    ],
    
    'Citation_Range': [
        '1-49 citations',
        '4-16 citations', 
        '2-63 citations',
        '2-234 citations',
        '2-36 citations',
        '14-16 citations',
        '105-234 citations',
        'New (2024-2025)',
        '1-36 citations'
    ],
    
    'Lending_Club_Usage': [
        'Yes (BP Neural Network, ANN comparison)',
        'Limited',
        'Some applications', 
        'Moderate usage',
        'High usage (BERT, GPT, Credit Risk BERT)',
        'Limited (mostly synthetic)',
        'High (Namvar 2018 - 170 citations)',
        'Emerging applications',
        'High (Generalist scoring, GPT applications)'
    ]
}

df_arch = pd.DataFrame(architecture_summary)
print("Neural Network Architecture Categories for Credit Risk Prediction")
print("="*70)
print(df_arch.to_string(index=False))

# Create a timeline analysis
timeline_data = {
    'Year': [2015, 2018, 2020, 2021, 2022, 2023, 2024, 2025],
    'Papers_Published': [1, 2, 3, 2, 6, 4, 4, 9],
    'Key_Developments': [
        'First neural network P2P study',
        'Imbalanced learning focus, SVM comparison',
        'Ensemble methods, Sequential deep learning, GANs', 
        'Deep ensemble models, BP networks',
        'CNN-LSTM hybrids, Multiple studies',
        'LLMs emergence, GANs synthesis, Multi-modal',
        'Transformer boom, BERT applications',
        'LLM explosion, Novel architectures, Systematic reviews'
    ]
}

df_timeline = pd.DataFrame(timeline_data)
print("\n\nTemporal Evolution of Credit Risk Neural Network Research")
print("="*60)
print(df_timeline.to_string(index=False))

# High-impact papers analysis
high_impact = {
    'Paper': [
        'Credit risk prediction in imbalanced social lending (Namvar 2018)',
        'A new deep learning ensemble credit risk evaluation (Shen 2021)', 
        'Multi-view ensemble learning for P2P credit risk (Song 2020)',
        'Sequential Deep Learning for Credit Risk Monitoring (Clements 2020)',
        'Machine learning and ANN for P2P credit scoring (Chang 2022)',
        'A credit risk assessment model with BP neural network (Ma 2021)',
        'Generalist Credit Scoring through LLMs (Feng 2023)'
    ],
    
    'Citations': [170, 234, 105, 63, 53, 49, 36],
    
    'Architecture': [
        'Random Forest + Resampling',
        'Deep Learning Ensemble', 
        'Multi-view Ensemble',
        'Temporal CNN',
        'ML + ANN methods',
        'BP Neural Network',
        'Large Language Models'
    ],
    
    'Key_Contribution': [
        'Systematic imbalanced learning for P2P, benchmark study',
        'Novel ensemble approach for imbalanced credit data',
        'Distance-to-model ensemble for P2P platforms', 
        'Sequential modeling with temporal convolution',
        'Comparative analysis of ML vs ANN methods',
        'BP network with LM algorithm for Chinese P2P',
        'First comprehensive LLM framework for credit scoring'
    ]
}

df_impact = pd.DataFrame(high_impact)
print("\n\nHigh-Impact Papers (>30 citations)")
print("="*50)
print(df_impact.to_string(index=False))

print(f"\n\nFiles created:")
print("1. credit_risk_neural_networks_research_papers.csv - Complete paper database")
print("2. Analysis shows 31 papers across 2015-2025")
print("3. Peak research activity in 2025 (9 papers) due to LLM emergence")
print("4. Highest cited: Namvar 2018 (170 cites), Shen 2021 (234 cites)")
print("5. Strong trend toward hybrid architectures and LLMs in recent years")