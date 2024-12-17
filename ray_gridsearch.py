#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: ray_gridsearch.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 6-06-2024 
#
#===============================================================================
#    DESCRIPTION:  
#    
#   DEPENDENCIES:  
#
#          USAGE: python ray_gridsearch.py 
#===============================================================================

#Semary NA, Ahmed W, Amin K, Pławiak P, Hammad M. Improving sentiment classification using a RoBERTa-based hybrid model. Front Hum Neurosci. 2023 Dec 7;17:1292010. doi: 10.3389/fnhum.2023.1292010. PMID: 38130432; PMCID: PMC10733963.

import torch
from torch.nn import CrossEntropyLoss, LSTM, Module, Linear, Dropout, LayerNorm
from torch import cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers.utils import logging
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast, DataCollatorForTokenClassification, get_scheduler

from accelerate import Accelerator
from datasets import Dataset, load_from_disk
import numpy as np
import evaluate

from functools import partial
import os
import tempfile
from pathlib import Path
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import matplotlib.pyplot as plt

print("Cuda is available : ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = load_from_disk('corpus/dataset/train')
dev_set = load_from_disk('corpus/dataset/dev')
test_set = load_from_disk('corpus/dataset/test')

tagset_label2id = {'O': 0,'B-ORG': 1, 'I-ORG': 2, 'B-MISC': 3, 'I-MISC': 4, 'B-PER': 5, 'I-PER': 6, 'B-LOC': 7, 'I-LOC': 8}
tagset_id2label = dict(zip(tagset_label2id.values(), tagset_label2id.keys()))

class CustomRobertaTokenClassWithLSTM(Module):
    def __init__(self, config, num_labels, hidden_dim, n_hidden_layers, lstm_dropout):
        super().__init__()
        self.num_labels = num_labels # number of classes to predict

        # layer 1
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # layer 2
        self.dropout = Dropout(0.3) # same shape as input
        
        # layer 3
        self.lstm = LSTM(config.hidden_size, hidden_dim, num_layers=n_hidden_layers, batch_first=True) # takes last hidden layer output from roberta

        self.lnorm = LayerNorm(hidden_dim)
        
        self.lstm_dropout = Dropout(lstm_dropout) # same shape as input
        
        self.fc = Linear(hidden_dim, num_labels)

        # Initialize weights and apply final processing
        self.roberta.post_init()
       
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None,
                output_hidden_states=None, labels=None, return_dict=True):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # output of last hidden layer
        sequence_output = self.dropout(sequence_output)
        sequence_output,_ = self.lstm(sequence_output) # LSTM Outputs: output, (h_n, c_n) -> we only need output for token classification
        sequence_output = self.lstm_dropout(sequence_output)
        sequence_output = self.lnorm(sequence_output)

        logits = self.fc(sequence_output)
      
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return (loss, logits)


tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
config = RobertaConfig.from_pretrained("roberta-base", return_dict=False)


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

#train_smaller = torch.utils.data.Subset(train_set, indices=range(int(len(train_set) * 0.50)))
#dev_smaller = torch.utils.data.Subset(dev_set, indices=range(int(len(dev_set) * 0.20)))


def load_train_dev(batch_size):
    
    train_dataloader = DataLoader(
        train_set,
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    dev_dataloader = DataLoader(
        dev_set,
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )

    return train_dataloader, dev_dataloader


def convert_ids_2_labels(model_predictions, true_labels):

    # go through each batch and ignore when id = -100 (padding token)
    true_labels = [
        [tagset_id2label[label_id] 
        for label_id in batch if label_id != -100 ] 
        for batch in true_labels ]
    
    model_predictions = [
        [tagset_id2label[pred] 
         for (pred, lab) in zip(prediction, label) if lab != -100 ]
        for prediction, label in zip(model_predictions, true_labels) ]
    
    return model_predictions, true_labels # both are same length


def compute_metrics(model_predictions):
    """ input : ŷ,  logits (log prob of model's prediction)
                y, true label of current prediction

                converts prediction ids to labels in string for computing scores
                
        returns dict with overall precision, recall, f1, accuracy
    """
    logits, true_labels = model_predictions

    predictions = np.argmax(logits, axis=1) # take the predicted tag with highest score

    return metric.compute(predictions=predictions, references=true_labels)



def train_model(model, optimizer, lr_scheduler, accelerator, train_dataloader):
    for batch in train_dataloader:
        batch = accelerator.prepare(batch)  
        
        outputs = model(**batch) 
        loss = outputs[0]
        
        print(f"Loss: {loss.item()}")
        
        accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad() 

        
def eval_model(model, accelerator, dev_dataloader):
    metric = evaluate.load('seqeval')
    
    for batch in dev_dataloader:
        batch = accelerator.prepare(batch)  
        
        with torch.no_grad():
            outputs = model(**batch)

        #print(f"Logits shape: {outputs.logits.shape}")
        
        predictions = outputs[1].argmax(dim=-1)
        print(predictions)
        labels = batch["labels"]
        #print("labels: ", labels)
        
        #print("preds shape : ", outputs.logits.shape)
        #print("labels shape : ", labels.shape)

        predictions_gathered = accelerator.gather(predictions).detach().cpu().clone().numpy()
        labels_gathered = accelerator.gather(labels).detach().cpu().clone().numpy()

        true_predictions, true_labels = convert_ids_2_labels(predictions_gathered, labels_gathered) 
        metric.add_batch(predictions=true_predictions, references=true_labels) 
        
    metrics = metric.compute()
    #print(metrics)

    return metrics['overall_accuracy']


def objective(params):

    ### accelerator ###
    accelerator = Accelerator(mixed_precision='fp16')

    hidden_dim, num_lstm_layers, dropout, batch_size, n_epochs = params['hidden_dim'], params['n_layers'], params['dropout'], params['batch_size'], params['n_epochs']

    train_dataloader, dev_dataloader = load_train_dev(batch_size)

    ### Initialize model ###
    model = CustomRobertaTokenClassWithLSTM(config, len(tagset_label2id), hidden_dim, num_lstm_layers, dropout).to(device)
    model = torch.compile(model)

    ### Optimizer choice and learning rate ###
    optimizer = AdamW(model.parameters(), lr=1e-5)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["lstm_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    ### Scheduler ###
    n_training_steps = n_epochs * len(train_dataloader)  

    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        name="linear",
        num_warmup_steps=0,
        num_training_steps=n_training_steps
    )

    model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader
    ) 

    for epoch in range(start_epoch, n_epochs):
        model.train()
        ### Training ###
        train_model(model, optimizer, lr_scheduler, accelerator, train_dataloader)
        
        model.eval()
        ### Evaluation ###
        accuracy = eval_model(model, accelerator, dev_dataloader)

        checkpoint_data = {
            "epoch": epoch,
            "lstm_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                { "accuracy": accuracy},
                checkpoint=checkpoint,
            )

    print("Finished Training")

def main(num_samples=1, max_num_epochs=1):
    
    config = {
        "dropout": tune.uniform(0, 1),
        "n_layers": tune.randint(1, 5),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "n_epochs": tune.randint(1, max_num_epochs),
        "batch_size": tune.choice([8, 16, 32]),
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(objective),
        resources_per_trial={"cpu": 10, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("accuracy", mode="max")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")


    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")

    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)
            
            with open("best_checkpoint_data.pkl", "wb") as fp:
                pickle.dump(best_checkpoint_data, fp)


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=6)


