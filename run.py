import os.path
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score
import random
from tools import *


def train(model: Module,                    # model
          tokenizer,                        # tokenizer
          train_dataloader: DataLoader,     # training dataloader
          valid_dataloader: DataLoader,     # validation dataloader
          optimizer: Optimizer,             # optimizer
          device,                           # the first device of GPU(s)
          writer: SummaryWriter,            # the writer to save log: loss, learning rate, metrics, sample
          args: dict
          ):
    curr_step = 0
    best_metric = args['best_metric']
    for epoch in range(args['epoch_s'], args['epoch_e']):
        for batch in tqdm(train_dataloader):
            curr_step += 1

            # Move data to GPUs
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            model.train()
            outputs = model(input_ids=batch['input_token_ids'],
                            attention_mask=batch['input_attn_mask'],
                            labels=batch['label'])
            loss = outputs.loss
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()

            # Backward Pass
            loss /= args['accumulation_steps']
            loss.backward()

            # Add summaries to TensorBoard
            writer.add_scalar('Train/Loss', loss, curr_step)

            # Update the optimizer
            if curr_step % args['accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Validate
            if curr_step % args['valid_interval'] == 0 and valid_dataloader.__len__():
                metrics = test(model, valid_dataloader, tokenizer, device)
                sample_number = random.randint(0, valid_dataloader.__len__() - 1)
                add_to_writer(metrics, writer, curr_step, sample_number)

            # Save checkpoint
            if curr_step % args['ckpt_interval'] == 0:
                path = os.path.join(args['log_dir'], f"model_{curr_step}.pth")
                state_dict = {'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'epoch': epoch,
                              'best_metric': best_metric}
                torch.save(state_dict, path)

        # Validate every epoch
        metrics = test(model, valid_dataloader, tokenizer, device)
        sample_number = random.randint(0, valid_dataloader.__len__() - 1)
        add_to_writer(metrics, writer, curr_step, sample_number)

        if metrics['metric'] > best_metric:
            best_metric = metrics['metric']
            path = os.path.join(args['log_dir'], f"ep_{epoch}_{best_metric}.pth")
            state_dict = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch,
                          'best_metric': best_metric}
            torch.save(state_dict, path)


@torch.no_grad()
def test(model: Module,                    # model
         dataloader: DataLoader,           # dataloader
         tokenizer,
         device,                           # the first device of GPU(s)
         args: dict
         ) -> dict:

    input_text = []
    label_pr = []
    label_gt = []
    loss = []

    model.eval()
    for batch in tqdm(dataloader):
        # Move data to GPU/CPU
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Log input text
        input_text += batch['input_text']

        # Predict labels
        outputs = model(input_ids=batch['input_token_ids'],
                        attention_mask=batch['input_attn_mask'],
                        labels=batch['label'])
        loss.append(outputs.loss.item())

        if args['task_name'] == 'cls':
            logits = outputs.logits  # [B, C]
            label_pr += torch.argmax(logits, dim=1).detach().cpu().tolist()
            label_gt += batch['label'].detach().cpu().tolist()

        elif args['task_name'] == 'seq2seq':
            label_token_ids = model.generate(batch['input_token_ids'])
            label_pr += [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in label_token_ids]
            label_gt += batch['label_text']

        else:
            raise KeyError("expected args['task_name'] is cls or seq2seq, but got other")

    # Calculate metrics
    if args['task_name'] == 'cls':
        metrics = accuracy_score(label_gt, label_pr)
    else:
        metrics = accuracy_score(label_gt, label_pr)

    return {'input_text': input_text,
            'prediction': label_pr,
            'label': label_gt,
            'loss': torch.tensor(loss).mean().item(),
            'metric': metrics}

