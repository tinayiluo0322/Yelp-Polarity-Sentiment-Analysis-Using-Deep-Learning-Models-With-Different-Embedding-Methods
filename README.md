# Yelp-Polarity-Sentiment-Analysis-Using-Deep-Learning-Models-With-Different-Embedding-Methods
### 1. Introduction
Sentiment classification is a fundamental task in natural language processing (NLP), with applications in customer feedback analysis, brand reputation monitoring, and automated content moderation. In this study, we implemented and compared multiple deep learning and traditional machine learning models for binary sentiment classification on the Yelp Polarity dataset. The objective was to evaluate how different architectures and embedding strategies impact classification performance. The models tested include Logistic Regression, Vanilla RNN, and LSTM, each paired with two embedding strategies: learnable embeddings (LE) and pretrained Word2Vec embeddings (W2V). Performance was assessed using accuracy, precision, recall, F1-score, and loss.

### 2. Background

#### 2.1 Model Selection
We selected three different model architectures with increasing complexity:
- **Logistic Regression (Baseline):** Serves as a fundamental benchmark. It transforms text into a numerical representation by averaging word embeddings and applies logistic regression for classification.
- **Vanilla RNN:** Processes input text as a sequence of word embeddings and captures temporal dependencies through recurrent connections. However, RNNs struggle with long-term dependencies due to vanishing gradients.
- **LSTM (Long Short-Term Memory):** An advanced form of RNN designed to handle long-range dependencies more effectively through its gating mechanisms. It is expected to outperform Vanilla RNN due to its ability to retain information across longer text sequences.

#### 2.2 Word Embedding Strategies
- **Learnable Embeddings (LE):** The model initializes an embedding layer with random weights and updates them during training. This allows the embeddings to be optimized for the specific dataset, potentially improving classification performance.
- **Pretrained Word2Vec Embeddings (W2V):** The embeddings are trained separately on the dataset before being used in the model. This strategy helps in leveraging semantic relationships between words, reducing the risk of overfitting and improving generalization.

### 3. Experiment Setup

#### 3.1 Dataset Preparation
- The **Yelp Polarity dataset** from Hugging Face was used.
- The text was tokenized and converted into sequences of word indices.
- Padding was applied to ensure uniform input length.
- The dataset was split into training, validation, and test sets.

#### 3.2 Model Implementation
Each model was implemented with two embedding versions:
- **Logistic Regression (LogReg_LE, LogReg_W2V)**
- **Vanilla RNN (RNN_LE, RNN_W2V)**
- **LSTM (LSTM_LE, LSTM_W2V)**

All models were trained using cross-entropy loss and the Adam optimizer. Performance was evaluated using accuracy, precision, recall, and F1-score.

### 4. Results and Analysis

#### 4.1 Performance Metrics
| Model        | Loss    | Accuracy | Precision | Recall  | F1-score |
|-------------|--------|----------|-----------|---------|----------|
| LSTM_W2V    | 0.1229 | 0.9521   | 0.9465    | 0.9584  | 0.9524   |
| LSTM_LE     | 0.2591 | 0.9393   | 0.9280    | 0.9526  | 0.9401   |
| LogReg_LE   | 0.2110 | 0.9312   | 0.9297    | 0.9331  | 0.9314   |
| LogReg_W2V  | 0.2816 | 0.8946   | 0.8987    | 0.8893  | 0.8940   |
| RNN_W2V     | 0.6258 | 0.6487   | 0.5992    | 0.8979  | 0.7188   |
| RNN_LE      | 0.6845 | 0.5244   | 0.5266    | 0.4822  | 0.5034   |

#### 4.2 Detailed Analysis
- **LSTM_W2V achieved the highest performance** with an accuracy of **95.21%** and an F1-score of **0.9524**. This suggests that the combination of LSTM's ability to retain long-term dependencies and Word2Vec’s pretrained semantic knowledge enhances sentiment classification significantly.
- **LSTM_LE performed slightly worse** but still achieved high accuracy (93.93%) and an F1-score of 0.9401. The difference indicates that pretrained Word2Vec embeddings provided a useful semantic advantage over randomly initialized embeddings.
- **Logistic Regression models performed well**, with **LogReg_LE** outperforming **LogReg_W2V** slightly. This suggests that for simpler models, allowing embeddings to be trained on the dataset may be more beneficial than using pretrained embeddings.
- **RNN models performed significantly worse**, especially **RNN_LE**, which had the lowest accuracy (52.44%). This confirms that simple RNNs struggle with longer sequences, and their lack of sophisticated gating mechanisms leads to ineffective learning.
- **RNN_W2V performed better than RNN_LE**, with a significant boost in recall (89.79%), but its lower precision indicates that it made more false positive classifications.

### 5. Conclusion

#### 5.1 Model Comparisons
- **LSTM is the best-performing model overall**, demonstrating its strength in capturing long-range dependencies and handling sequential data effectively.
- **Logistic Regression provides a strong baseline** with relatively high accuracy, making it an efficient choice when computational resources are limited.
- **Vanilla RNN is the weakest model**, confirming that simple RNNs struggle with long text sequences and suffer from vanishing gradient problems.

#### 5.2 Embedding Strategy Comparisons
- **Word2Vec embeddings improved performance for complex models like LSTM and RNN** due to their ability to retain semantic relationships.
- **Learnable embeddings worked better for Logistic Regression**, suggesting that training embeddings from scratch is beneficial for simpler models without sequential dependencies. However, this approach comes at the cost of significantly increased training time.

#### 5.3 Final Thoughts
- **Best Choice:** LSTM with Word2Vec (LSTM_W2V) due to its superior accuracy, F1-score, and overall robustness.
- **Efficient Alternative:** Logistic Regression with learnable embeddings (LogReg_LE).
- **Least Recommended:** Vanilla RNN, as it struggled significantly, especially without pretrained embeddings.

This study highlights the importance of both model selection and embedding strategy in NLP tasks. Future work could explore Transformer-based models like BERT to further enhance performance.
