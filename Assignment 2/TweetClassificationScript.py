from absl import app, flags
import pandas as pd
import numpy as np
import random
import time
import datetime
import argparse
import torch
import os
from transformers import (get_linear_schedule_with_warmup, AdamW, AutoTokenizer, AutoModelForSequenceClassification)
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)

flags.DEFINE_integer('max_seq_length', 128, '')
flags.DEFINE_float('lr', 5e-5, '')

FLAGS = flags.FLAGS

MODEL_CLASSES = {
    'bertweet': 'vinai/bertweet-base',
    'roberta': 'cardiffnlp/twitter-roberta-base',
    'bert': 'bert-base-uncased'
}


def prepare_data(root_dir='.', dataset='hate'):
    if not os.path.exists(root_dir):
        return None
    # reading data in type list
    train_text = open(f'{root_dir}/datasets/{dataset}/train_text.txt', 'r', encoding="utf-8").readlines()
    train_labels = open(f'{root_dir}/datasets/{dataset}/train_labels.txt', 'r').readlines()
    val_text = open(f'{root_dir}/datasets/{dataset}/val_text.txt', 'r', encoding="utf-8").readlines()
    val_labels = open(f'{root_dir}/datasets/{dataset}/val_labels.txt', 'r').readlines()
    test_text = open(f'{root_dir}/datasets/{dataset}/test_text.txt', 'r', encoding="utf-8").readlines()
    test_labels = open(f'{root_dir}/datasets/{dataset}/test_labels.txt', 'r').readlines()

    # creating dataframe
    train_df = pd.DataFrame({'tweet': train_text, 'label': [int(lbl.strip('\n')) for lbl in train_labels]})
    val_df = pd.DataFrame({'tweet': train_text, 'label': [int(lbl.strip('\n')) for lbl in train_labels]})
    test_df = pd.DataFrame({'tweet': train_text, 'label': [int(lbl.strip('\n')) for lbl in train_labels]})
    num_classes = train_df['label'].unique().shape[0]
    return train_df, val_df, test_df, num_classes


def encode(df, tokenizer, max_seq_length=512):
    input_ids = []
    attention_masks = []
    for tweet in df[["tweet"]].values:
        tweet = tweet.item()
        encoded_dict = tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=FLAGS.max_sequence_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {
        'input_word_ids': input_ids,
        'input_mask': attention_masks}

    return inputs


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def data_loader(train_df, test_df, val_df, batch_size, tokenizer_class="vinai/bertweet-base"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_class, use_fast=False, normalization=True)

    train = encode(train_df, tokenizer)
    train_labels = train_df['label'].astype(int)
    validation = encode(val_df, tokenizer)
    validation_labels = val_df['label'].astype(int)
    test = encode(test_df, tokenizer)
    test_labels = test_df['label'].astype(int)

    input_ids, attention_masks = train.values()
    labels = torch.tensor(train_labels.values)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    input_ids, attention_masks = validation.values()
    labels = torch.tensor(validation_labels.values)
    val_dataset = TensorDataset(input_ids, attention_masks, labels)
    input_ids, attention_masks = test.values()
    labels = torch.tensor(test_labels.values)
    test_dataset = TensorDataset(input_ids, attention_masks, labels)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    return train_dataloader, validation_dataloader, test_dataloader


def prepare_model(model_class="vinai/bertweet-base", num_classes=2, model_to_load=None, total_steps=-1):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_class,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    )

    optimizer = AdamW(model.parameters(),
                      lr=FLAGS.lr,
                      eps=1e-8
                      )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    if model_to_load is not None:
        try:
            model.roberta.load_state_dict(torch.load(model_to_load))
            print("LOADED MODEL")
        except:
            pass
    return model, optimizer, scheduler


def predict(model, test_dataloader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    predictions = []
    t0 = time.time()
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    prediction_time = format_time(time.time() - t0)
    print("  Test took: {:}".format(prediction_time))
    return predictions


def validate(model, val_dataloader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    total_eval_accuracy = 0
    total_eval_loss = 0
    t0 = time.time()
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("  Accuracy: {0:.2f} %".format(avg_val_accuracy * 100))
    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)
    print("  Test Loss: {0:.2f}".format(avg_val_loss))
    print("  Test took: {:}".format(validation_time))
    return avg_val_accuracy, avg_val_loss, validation_time


def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, save_location):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        print("")
        print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        avg_val_accuracy, avg_val_loss, validation_time = validate(model, validation_dataloader)
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    torch.save(model.cpu().roberta.state_dict(), save_location)


def save_predictions(predictions, task):
    preds = [pl for batch in predictions for pl in np.argmax(batch, axis=1)]
    with open(f'predictions/{task}.txt', 'r+') as f:
        f.seek(0)
        for item in preds:
            f.write("%s\n" % item)
        f.truncate()
    print("Predictions saved!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size for training')
    parser.add_argument('--epochs', type=int, default=4, help='The batch size for training')
    parser.add_argument('--dataset_root_path', type=str, default='.', help='Root directory path to the dataset')
    parser.add_argument('--model_class', type=str, default='bertweet', choices=['bertweet', 'bert', 'roberta'],
                        help='The pre-trained hugginface model to load')
    parser.add_argument('--task', type=str, default='sentiment', choices=['emotion', 'hate', 'sentiment'],
                        help='The TweetEval dataset to choose')
    parser.add_argument('--model_to_load', type=str, default=None, help='Load pre-trained BERT')
    parser.add_argument('--save', type=str, default='./model.pb', help='Save the model to disk')

    args = parser.parse_args()

    train_df, val_df, test_df, num_classes = prepare_data(args.dataset_root_path, args.task)
    train_dataloader, validation_dataloader, test_dataloader = data_loader(train_df, test_df, val_df,
                                                                           tokenizer_class="vinai/bertweet-base",
                                                                           batch_size=args.batch_size)
    total_steps = len(train_dataloader) * args.epochs
    model, optimizer, scheduler = prepare_model(MODEL_CLASSES[args.model_class], num_classes, args.model_to_load,
                                                   total_steps)
    train(model, optimizer, scheduler, train_dataloader, validation_dataloader, args.epochs, args.save)
    predictions = predict(model, test_dataloader)
    save_predictions(predictions, args.task)


if __name__ == '__main__':
    app.run(main)
