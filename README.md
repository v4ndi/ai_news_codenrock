# AI_NEWS solution

## Description
News Classification and Duplicate Detection  
Our project involves the classification of news articles into 29 distinct categories and the identification of duplicate articles. To achieve this, we have developed a classification model based on the pre-trained `ai-forever/ruRoberta-large` model from Hugging Face. Duplicate detection is performed using TF-IDF and cosine distance measurements between embeddings, with a threshold set at >= 0.95.

Additionally, we have implemented a FAST API for easy access and interaction with our model. You can find the fine-tuning notebook for our model at additional/FinTune_ruRoberta_large.ipynb.
