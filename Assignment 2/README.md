# Resolving Mysteries of Twitter Data (Deeper Dive)
## Description
I have used three transformer based models (BERT variants) from [huggingface](https://huggingface.co/) in this project and have evaluated their performances with [TweetEval benchmark](https://github.com/cardiffnlp/tweeteval) using different transfer learning scenarios.
* [BERTweet](https://www.aclweb.org/anthology/2020.emnlp-demos.2/) - The corpus used to pre-train BERTweet consists of 850M English Tweets (16B word tokens ~ 80GB), containing 845M Tweets streamed from 01/2012 to 08/2019 and 5M Tweets related to the COVID-19 pandemic.
* [Roberta-twitter](https://arxiv.org/abs/2010.12421) - roBERTa-base model pre-trained on ~58M tweets. Twitter masked language model RoBERTa-retrained (task-specific fine-tuned) is also available in huggingface for you to download and evaluate.
* [BERT-base-uncased](https://github.com/google-research/bert) - Trained on lower-cased English text, available with 12 encoder layers, 768 output hidden-size, 12-heads, 110M parameters.

## Tasks & Dataset:  
This project is built for three tasks (Sentiment Analysis, Emotion Recognition, and Hate Speech Detection) with the task-specific datasets from TweetEval Framework. To download the dataset: 
```bash
git clone https://github.com/cardiffnlp/tweeteval
```
## Results

| Model | Sentiment [1] | Emotion [2] | Hate [3] |
|----------|------:|--------:|-----:|
| BERTweet   | 70.79       | 85.18       | 59.06    |
| RoBERTa-base  | 70.91      | 84.35       | 56.76   |
| BERT-base | **70.95**     | 84.59       | 56.12    |

## Dependancies
To install dependancies run the following command:
```bash
pip install -r requirements.txt
```

## Training:
Here is the syntax of the python file
```
usage: TweetClassificationScript.py [-h] [--batch_size BATCH_SIZE]
                                               [--epochs EPOCHS]
                                               [--total_steps TOTAL_STEPS]
                                               [--dataset_location DATASET_LOCATION]
                                               [--model_class MODEL_CLASS]
                                               [--dataset {emotion,hate,sentiment}]
                                               [--model_to_load MODEL_TO_LOAD]
                                               [--save SAVE]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        The batch size for training
  --epochs EPOCHS       The batch size for training
  --total_steps TOTAL_STEPS
                        Number of training steps
  --dataset_location DATASET_LOCATION
                        The tweetEval dataset location
  --model_class MODEL_CLASS
                        The pre-trained hugginface model to load
  --dataset {emotion,hate,sentiment}
                        The TweetEval dataset to choose
  --model_to_load MODEL_TO_LOAD
                        Load pre-trained BERT
  --save SAVE           Save the model to disk
```

Model is by default [vinai/bertweet-base](https://huggingface.co/vinai/bertweet-base)

## Evaluating the system
For evaluating the system and the predictions made by the script, you simply need a predictions file for each of the tasks in the predictions folder. The format of the predictions file should be the same as the output examples in the predictions folder (one output label per line as per the original test file). The best predictions made by the script with BERTweet-base model, are included as an example in this repo.

#### usage
```bash
python evaluation_script.py
```
The script takes the TweetEval gold/actual test labels and the predictions from the "predictions" folder by default, but you can set this to suit your needs as optional arguments.

Optional arguments
Three optional arguments can be modified:

_--tweeteval_path_: Path to TweetEval datasets. Default: "./datasets/"

_--predictions_path_: Path to predictions directory. Default: "./predictions/"

_--task_: Use this to get single task detailed results (emotion|hate|sentiment). Default: ""

Evaluation script sample usage from the terminal with parameters:

```bash
python evaluation_script.py --tweeteval_path ./datasets/ --predictions_path ./predictions/ --task emoji
```
