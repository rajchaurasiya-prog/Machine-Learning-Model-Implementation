Name:Raj Chaurasiya

Intern Id:CT04DZ1762

Domain Name:Python programming

Duration:4 weeks

Mentor Name:Neela Santosh

Description:

The given Python program is a spam message classifier using Natural Language Processing (NLP) techniques and the Naive Bayes algorithm from the scikit-learn library. It is designed to identify whether a given text message (SMS) is spam or not spam (also known as ham). The dataset used in this example is a publicly available file containing labeled SMS messages, and the overall goal is to train a machine learning model that can accurately classify new, unseen messages.

Step-by-step Description
1. Importing Libraries
The script begins by importing essential libraries:
pandas and numpy for data manipulation,
matplotlib.pyplot and seaborn for visualization,
sklearn for machine learning tasks such as model training, evaluation, and data preprocessing.

2. Loading the Dataset
The dataset is imported directly from a URL using pandas.read_csv() with tab-separated values (sep="\t"). It contains two columns:
"label": either "ham" (not spam) or "spam", and
"message": the text content of the SMS.
This dataset is widely used for NLP and spam filtering demonstrations.

3. Visualizing the Data
The script visualizes the distribution of spam vs ham messages using a bar chart. This is an important step in understanding whether the dataset is balanced (equal numbers of spam and ham messages). Imbalanced data can bias the machine learning model toward the majority class.

4. Label Encoding
Machine learning models cannot directly work with categorical values like "ham" and "spam", so these labels are converted to binary numeric values:
"ham" becomes 0, and
"spam" becomes 1.
This transformation is necessary for model training.

5. Text Vectorization
Since machine learning models require numeric input, the raw SMS messages (strings) are transformed using CountVectorizer. This method converts a collection of text documents into a matrix of token counts. Each word in the corpus is represented as a feature, and each message is converted into a vector representing the frequency of each word.

6. Training and Testing Split
The dataset is split into training and testing sets using train_test_split(). Here, 80% of the data is used for training the model, and 20% is reserved for testing. The random_state ensures reproducibility.

7. Model Training
A Multinomial Naive Bayes (MNB) classifier is used, which is particularly suitable for classification with discrete features like word counts. The model is trained using the training data (X_train, y_train).

8. Model Evaluation
After training, predictions are made on the test data (X_test), and the model’s performance is evaluated using:
Accuracy score (percentage of correctly predicted messages),
Classification report (precision, recall, F1-score), and
Confusion matrix visualized as a heatmap using Seaborn.
This comprehensive evaluation helps assess the model's effectiveness in detecting spam.

9. Prediction Function
Finally, a user-defined function predict_message(msg) allows predicting new, unseen SMS messages. The input message is vectorized and classified using the trained model. In the provided example, a spam-like message (“You have won $1000…”) is correctly identified as spam.

Conclusion
This script is a practical implementation of spam detection using classical NLP and machine learning methods. It demonstrates data loading, preprocessing, model training, evaluation, and prediction. It can be further improved by incorporating more advanced techniques like TF-IDF vectorization, stop-word removal, stemming, or even using deep learning models for improved performance on larger datasets.




output:

![Image](https://github.com/user-attachments/assets/024b5fd3-f4a2-49fc-902c-90b13da51361)
