# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from joblib import Memory
# from transformers import BertForQuestionAnswering, BertForTokenClassification


def prepare_input_data(idir, bert_name):
    print('load from', idir)
    pos_texts = pd.read_csv(f'{idir}/positive.csv', encoding='utf-8', sep=';', header=None)
    neg_texts = pd.read_csv(f'{idir}/negative.csv', encoding='utf-8', sep=';', header=None)
    neg_texts.head(10)

    sentences = np.concatenate([pos_texts[3].values, neg_texts[3].values])
    sentences = ['[CLS] ' + s + '[SEP]' for s in sentences]
    labels = [[1] for _ in range(pos_texts.shape[0] + neg_texts.shape[0])]

    assert len(sentences) == len(labels) == pos_texts.shape[0] + neg_texts.shape[0]

    print(sentences[1000])
    train_sentences, test_sentences, train_gt, test_gt = train_test_split(sentences, labels, test_size=0.3)
    print('train test len:', len(train_gt), len(test_gt))

    tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(s) for s in train_sentences]
    print(tokenized_texts[0])

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(
        input_ids,
        maxlen=100,
        dtype='long',
        truncating='post',
        padding='post',
    )

    attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]

    # prediction_inputs = torch.tensor(input_ids)
    # prediction_masks = torch.tensor(attention_masks)
    # prediction_labels = torch.tensor(test_gt)

    # prediction_data = TensorDataset(
    #     prediction_inputs, prediction_masks, prediction_labels
    # )
    #
    # prediction_dataloader = DataLoader(
    #     prediction_data, sampler=SequentialSampler(prediction_data), batch_size=32
    # )

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, train_gt,
        random_state=42,
        test_size=0.1
    )
    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks,
        input_ids,
        random_state=42,
        test_size=0.1
    )

    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=12,
    )

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_dataloader = DataLoader(
        validation_data,
        sampler=RandomSampler(validation_data),
        batch_size=12,
    )
    return train_dataloader, validation_dataloader


def train_model(train_dataloader,  bert_name, device):
    model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=2)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    train_loss_set = []
    train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss[0].item())
        # backward pass
        loss[0].backward()
        # update params
        optimizer.step()
        train_loss += loss[0].item()

        # plot
    #     plt.plot(train_loss_set)
    #     plt.title('Training loss')
    #     plt.xlabel('Batch')
    #     plt.ylabel('Loss')
    #
    # plt.show()

    print(f'loss train: {train_loss / len(train_dataloader):.5f}')
    return model


def estimate_model(model, validation_dataloader):
    valid_preds, valid_labels = [], []
    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = logits[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        batch_preds = np.argmax(logits, axis=1)
        batch_labels = np.concatenate(label_ids)
        valid_preds.extend(batch_preds)
        valid_labels.extend(batch_labels)
    score = accuracy_score(valid_labels, valid_preds)
    print(f'right answers percent val: {score:.3f}')
    return score


def main():
    idir = 'data/tweet_sent'
    location = './cache'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_name = 'bert-base-uncased'
    memory = Memory(location, verbose=0)

    train_dataloader, validation_dataloader = memory.cache(prepare_input_data)(idir, bert_name)
    model = memory.cache(train_model)(train_dataloader,  bert_name, device)
    model.eval()
    estimate_model(model, validation_dataloader)


if __name__ == '__main__':
    main()
