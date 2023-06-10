import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from datasets import my_collate_bert
from transformers import AdamW
from utils import calculate_metrics


def initialize_random_seed(args):
    """
    Initializes random seed for reproducibility.
    
    Args:
        args: Command line arguments parsed by ArgumentParser.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def extract_batch_inputs(args, batch):
    """
    Extracts inputs and labels from a batch.
    
    Args:
        args: Command line arguments parsed by ArgumentParser.
        batch: The current batch of data.
        
    Returns:
        Tuple with inputs and labels.
    """
    inputs = {
        'input_ids': batch[0],
        'input_aspect_ids': batch[2],
        'word_indexer': batch[1],
        'aspect_indexer': batch[3],
        'input_cat_ids': batch[4],
        'segment_ids': batch[5],
        'dep_tags': batch[6],
        'pos_class': batch[7],
        'text_len': batch[8],
        'aspect_len': batch[9],
        'dep_rels': batch[11],
        'dep_heads': batch[12],
        'aspect_position': batch[13],
        'dep_dirs': batch[14]
    }
    labels = batch[10]
    return inputs, labels


def get_collate_function(args):
    """
    Returns the appropriate collate function for the DataLoader.
    
    Args:
        args: Command line arguments parsed by ArgumentParser.
    
    Returns:
        The collate function.
    """
    return my_collate_bert


def configure_bert_optimizer(args, model):
    """
    Configures the optimizer for the model.
    
    Args:
        args: Command line arguments parsed by ArgumentParser.
        model: The model to be optimized.
        
    Returns:
        The configured optimizer.
    """
    # Prepare optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer


def train_model(args, train_dataset, model, test_dataset):
    """
    Trains the model and evaluates it at certain intervals.
    
    Args:
        args: Command line arguments parsed by ArgumentParser.
        train_dataset: The dataset to train the model on.
        model: The model to be trained.
        test_dataset: The dataset to evaluate the model on.
        
    Returns:
        The global training step, training loss, and all evaluation results.
    """

    # Update training batch size
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_function(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    # Calculate total steps for training
    if args.max_steps > 0:
        total_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        total_steps = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = configure_bert_optimizer(args, model)

    # Training loop
    print("***** Running training *****")
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    initialize_random_seed(args)
    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = extract_batch_inputs(args, batch)
            logits = model(**inputs)
            loss = F.cross_entropy(logits, labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate_model(args, test_dataset, model)
                    all_eval_results.append(results)
                    for key, value in results.items():
                        print('eval_{}'.format(key), value, global_step)
                    print('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                if args.max_steps > 0 and global_step > args.max_steps:
                    break
        if args.max_steps > 0 and global_step > args.max_steps:
            break
    return global_step, tr_loss/global_step, all_eval_results


def evaluate_model(args, eval_dataset, model):
    """
    Evaluates a model with the given dataset.
    
    Args:
        args: Command line arguments parsed by ArgumentParser.
        eval_dataset: The dataset to evaluate the model on.
        model: The model to be evaluated.
        
    Returns:
        The results of the evaluation and the evaluation loss.
    """
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_function(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    print("***** Running evaluation *****")
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = extract_batch_inputs(args, batch)
            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = calculate_metrics(preds, out_label_ids)
    results.update(result)
    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(output_eval_file, 'a+') as writer:
        print('***** Eval results *****')
        print("eval loss: {}".format(eval_loss))
        for key in sorted(result.keys()):
            print("{} = {}".format(key, result[key]))

    return results, eval_loss


def evaluate_bad_cases(args, eval_dataset, model, word_vocab):
    """
    Evaluates model on cases it performs poorly.
    
    Args:
        args: Command line arguments parsed by ArgumentParser.
        eval_dataset: The dataset to evaluate the model on.
        model: The model to be evaluated.
        word_vocab: Vocabulary of words.
        
    Returns:
        A list of bad cases where the model prediction doesn't match the label.
    """
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_function(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)
    bad_cases = []
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = extract_batch_inputs(args, batch)
            logits = model(**inputs)
        pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        label = int(labels.detach().cpu().numpy()[0])
        if pred != label:
            sent_ids = inputs['input_ids'][0].detach().cpu().numpy()
            aspect_ids = inputs['input_aspect_ids'][0].detach().cpu().numpy()
            case = {
                'sentence': args.tokenizer.decode(sent_ids),
                'aspect': args.tokenizer.decode(aspect_ids),
                'pred': pred,
                'label': label
            }
            bad_cases.append(case)
    return bad_cases
