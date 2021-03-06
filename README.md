# A Comprehensive Evaluation on Tweets Classification

This is the repository for [Resolving Mysteries of Twitter Data](https://drive.google.com/file/d/1h_x1vnom1Av1Y2rQV5OTlJuuRrusXfKx/view?usp=sharing). This project aims to achieve three heterogeneous tasks in Twitter from [TweetEval Evaluation Framework](https://github.com/cardiffnlp/tweeteval), all framed as multi-class tweet classification. Datasets for each task from TweetEval is in the same format and with fixed training, validation and test splits.

## Datasets
This repository contains seven datasets for seven different tasks. However, as stated above the notebook here will be implemented on three tasks and their respective datasets.

Multi-class classification tasks in this project scope:
- [x] Hate Speech Detection: [SemEval 2019 (Hateval)](https://www.aclweb.org/anthology/S19-2007.pdf) - 2 labels: hateful, not hateful
- [x] Emotion Recognition: [SemEval 2018 (Emotion Recognition)](https://www.aclweb.org/anthology/S18-1001/) - 4 labels: anger, joy, sadness, optimism
- [x] Sentiment Analysis: [SemEval 2017 (Sentiment Analysis in Twitter)](https://www.aclweb.org/anthology/S17-2088/) - 3 labels: positive, neutral, negative

Other available multi-class classification tasks (not in this project scope):
- [] Emoji Prediction, SemEval 2018 (Emoji Prediction) - 20 labels: ❤️, 😍, 😂 ... 🌲, 📷, 😜
- [] Irony Detection, SemEval 2018 (Irony Detection) - 2 labels: irony, not irony
- [] Offensive Language Identification, SemEval 2019 (OffensEval) - 2 labels: offensive, not offensive
- [] Stance Detection*, SemEval 2016 (Detecting Stance in Tweets) - 3 labels: favour, neutral, against

## Pre-trained model and code
You can download the best Twitter masked language model (RoBERTa-retrained in the paper) from 🤗HuggingFace here, that is being used in the notebook for Sentiment Analysis:

Twitter-RoBERTa-sentiment
