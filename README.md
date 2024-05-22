# Stock-price-prediction-using-LSTM-and-Sentiment-Analysis
The program forecasts stock prices by combining sentiment analysis with an LSTM neural network. Sentiment scores derived from BERT and VADER, analyzing financial news and social media, are integrated with historical data of Apple Inc., etc. This enriched dataset feeds into a Keras-built LSTM model.

**1	Introduction**

Stock price prediction is crucial for financial decision-making, guiding investors to discern market trends and execute informed investments. The integration of machine learning with sentiment analysis has enhanced the modeling of complex financial data. This project employs Long Short-Term Memory (LSTM) neural networks, combined with sentiment analysis using BERT and VADER, to predict stock prices. Focusing on Apple Inc., known for its volatile stock performance, we leverage LSTM's ability to capture temporal dependencies and sentiment analysis's insight into market mood. The model, trained on historical and sentiment data, aims to accurately forecast Apple's stock prices. Its efficacy is further tested on Nvidia Corporation, evaluating the model's generalization across different stocks. This dual approach, merging quantitative analysis with sentiment metrics, aims to demonstrate the enhanced predictive capability of LSTM networks in financial forecasting.

**2	Problem Specification**

The primary goal is to develop a robust predictive model for stock price forecasting, leveraging Long Short-Term Memory (LSTM) neural networks and integrating sentiment analysis. The model is designed to accurately predict the future stock prices of Apple Inc., utilizing not only historical stock price data but also sentiment scores derived from BERT and VADER analyses of financial news and social media. Furthermore, the model's architecture must be carefully designed, incorporating multiple LSTM layers and appropriate hyperparameters. The choice of optimizer, such as Adam, plays a crucial role in optimizing the model's performance during training. Subsequently, the trained model's efficacy is evaluated by testing its predictions against unseen data from Nvidia Corporation, another technology giant. The key challenge lies in developing a model that not only achieves high accuracy in predicting Apple's stock prices but also demonstrates robust generalization capabilities to predict stock prices for a different company. Hence, the problem specification encompasses data preprocessing, model training, evaluation, and validation to address these objectives effectively.

**3	Design Details**

Our design integrates two sentiment analysis methods, BERT and VADER, to compute comprehensive sentiment scores. BERT, with its deep learning capabilities, excels in understanding context and nuances in large datasets, while VADER, a lexicon and rule-based model, efficiently handles sentiment scoring, especially for shorter texts and social media content. By merging these sentiment scores, we obtain a nuanced view of market sentiment. These scores are then combined with historical stock price data to feed into an LSTM (Long Short-Term Memory) network, which is adept at handling time series prediction. The LSTM model analyzes the temporal patterns and relationships between sentiment and stock price movements, enabling us to make informed predictions about future stock prices. This design leverages the strengths of both sentiment analysis and time series modeling to provide a sophisticated approach to predicting stock market trends.

**3.1 Sentiment Analysis**

3.1.1 Using BERT

BERT (Bidirectional Encoder Representations from Transformers) is used for our sentiment analysis because it leverages the power of transfer learning, which significantly enhances the model's ability to understand and process natural language. Using BERT with transfer learning reduces the need for large domain-specific datasets and computational resources for training from scratch.
Implementing BERT for sentiment analysis in the context of predicting stock prices begins with the collection and preprocessing the financial news. Our training data comes from Yahoo-Finance-News-Sentences on HuggingFace [1]. 

The next step is tokenization and input representation, where the preprocessed text is broken down into tokens using BERT's WordPiece tokenization method. Following tokenization, the BERT model undergoes a fine-tuning process. Although BERT comes pre-trained on a vast corpus of data, fine-tuning it on domain-specific financial data allows the model to adapt to the nuances of financial language and sentiment. This is achieved by training the model on a labeled dataset where texts are associated with sentiment labels, enabling BERT to learn and understand the context and sentiment specific to financial texts.
After fine-tuning, the model is used for sentiment classification. This involves adding a classification layer to the top of the BERT model, which utilizes the feature vectors produced by BERT to determine the sentiment expressed in the text. The output from this stage is a sentiment score that reflects the overall sentiment of the text, ranging from positive to negative.

3.1.2 Using VADER

We also use VADER to calculate the sentiment score. VADER assigns each piece of text a compound sentiment score, which quantifies the overall sentiment on a scale from negative to positive. This scoring system is particularly useful in financial sentiment analysis, where the market's reaction to news can be immediate and impactful.
By incorporating both immediate, lexicon-based sentiment assessments and the contextual, deep learning-driven insights from BERT, our design achieves a balanced and comprehensive approach to sentiment analysis in the financial domain.
A correlation analysis will be conducted to examine the extent to which the stock price is related to, and influenced by, the sentiment score. This analysis aims to quantify the relationship between market sentiment and stock price movements, providing insight into how sentiment scores correlate with stock price changes.

**3.2 Time Series Analysis**

The dataset utilized in this project comprises historical stock price data obtained from Yahoo Finance[2]. It encompasses various features such as Date, Open, High, Low, Close, Adjusted Close, and Volume. However, for the purpose of this study, only the Adjusted Close feature, denoted as 'Adj Close,' is utilized as the primary data for analysis. This feature serves as a reliable indicator of the stock's closing price after adjustments for factors such as dividends, stock splits, and other corporate actions [3].

The data preprocessing phase involves preparing the dataset for model training and testing. Each data point is transformed into a dataset matrix using a sliding window approach. Specifically, the 'create_dataset' function segments the data into input-output pairs. Each input sequence comprises a sequence of historically adjusted closing prices, while the corresponding output represents the next adjusted closing price in the sequence. This process enables the model to learn from past stock price trends and patterns.

The LSTM neural network model, constructed using the Keras library, adopts a sequential architecture. The model architecture consists of multiple LSTM layers, each with a specific number of LSTM units. These layers are designed to capture temporal dependencies and intricate patterns within the time series data. Additionally, a dense output layer is included to produce the final predictions based on the learned features extracted by the LSTM layers. The choice of the Adam optimizer is made due to its effectiveness in optimizing the model parameters during the training process.

The model training phase involves fitting the LSTM model to the prepared training dataset. The training process entails multiple epochs, during which the model learns to minimize the Mean Squared Error (MSE) loss between the predicted and actual stock prices. The training progress is monitored using a subset of the data reserved for validation, allowing for the evaluation of the model's performance on unseen data. This validation dataset aids in preventing overfitting and ensuring the model's generalization capability.

For evaluating the model's performance, a broader assessment is conducted by testing the model on stock price data from various companies, each representing different market segments. In addition to Apple Inc., which serves as a benchmark, the model is tested on stock prices from Nvidia Corporation, representing another prominent technology company. Furthermore, the evaluation extends to include stock prices from Metro, a lesser-known company, and Globalstar, a Micro-cap stock. This diversified approach aims to gauge the model's effectiveness across different market scenarios and company profiles. By testing on a range of companies with varying market capitalizations and industry sectors, the evaluation provides valuable insights into the model's generalization capability and robustness. This comprehensive assessment enables stakeholders to assess the model's performance across diverse market conditions and make informed decisions based on its predictive accuracy.

Finally, the trained model is utilized to forecast future stock prices for Apple Inc. The model's predictive capabilities are demonstrated by projecting the next 30 days' adjusted closing prices, starting from a specific date. This forecasting exercise provides stakeholders with valuable insights into potential future trends in the stock market, enabling informed decision-making and strategic planning.



**References**

[1] Sadiklar, U. (2024) UGURSA/Yahoo-finance-news-sentences · datasets at hugging face, ugursa/Yahoo-Finance-News-Sentences · Datasets at Hugging Face. Available at: https://huggingface.co/datasets/ugursa/Yahoo-Finance-News-Sentences.

[2] Yahoo Finance. (2024). Apple Inc. historical data from 2011/01/03 - 2017/12/29. Retrieved from https://finance.yahoo.com/quote/AAPL/history?period1=1294012800&period2=1514505600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

[3] Ganti, A. (2020), “Adjusted Closing Price: How It works, Types, Pros, & Cons”,  available at: https://www.investopedia.com/terms/a/adjusted_closing_price.asp

