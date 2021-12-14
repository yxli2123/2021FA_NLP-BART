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
from rouge import Rouge


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
                metrics = test(model, valid_dataloader, tokenizer, device, args)
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
        metrics = test(model, valid_dataloader, tokenizer, device, args)
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
        acc = 100 * accuracy_score(label_gt, label_pr)
        r1 = acc
        r2 = acc
        rl = acc
    else:
        rouge = Rouge()
        # don't allow emypty string when compute ROUGE
        label_gt_new = []
        label_pr_new = []
        for str_gt, str_pr in zip(label_gt, label_pr):
            if str_gt == "" or str_pr == "":
                continue
            else:
                label_gt_new.append(str_gt)
                label_pr_new.append(str_pr)

        metrics = rouge.get_scores(label_gt_new, label_pr_new, avg=True)
        r1 = 100 * metrics['rouge-1']['r']
        r2 = 100 * metrics['rouge-2']['r']
        rl = 100 * metrics['rouge-l']['r']
        acc = (r1 + r2 + rl) / 3
    print("r1: ", r1)
    print("r2: ", r2)
    print("rl: ", rl)
    return {'input_text': input_text,
            'prediction': label_pr,
            'label': label_gt,
            'loss': torch.tensor(loss).mean().item(),
            'metric': acc,
            'metrics': {'r1': r1, 'r2': r2, 'rl': rl}}

