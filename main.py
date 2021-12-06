from model import get_model
import argparse
from tools import *
from dataloader import BaseDataset
from model import get_model
from torch.utils.data import DataLoader
from torch import optim
from run import *
from torch.utils.tensorboard import SummaryWriter
import os


def config():
    parser = argparse.ArgumentParser(description='Reproduce BART')

    # Experiment params
    parser.add_argument('--mode',               type=str,       required=True,  choices=['train', 'test'])
    parser.add_argument('--expt_dir',           type=str)
    parser.add_argument('--expt_name',          type=str)
    parser.add_argument('--run_name',           type=str)

    # Model params
    parser.add_argument('--model',              type=str,       required=True)
    parser.add_argument('--seq_len',            type=int,       default=1024)
    parser.add_argument('--num_cls',            type=int,       default=3)
    parser.add_argument('--ckpt',               type=str)

    # Data params
    parser.add_argument('--dataset',            type=str,       required=True,  choices=['cnn_dailymail', 'multi_nli'])

    # Training params
    parser.add_argument('--seed',               type=int,       default=888)
    parser.add_argument('--lr',                 type=float,     default=1e-5)
    parser.add_argument('--epochs',             type=int,       default=10)
    parser.add_argument('--batch_size',         type=int,       default=8)
    parser.add_argument('--acc_step',           type=int,       default=1)
    parser.add_argument('--valid_interval',     type=int,       default=1000)
    parser.add_argument('--save_interval',      type=int,       default=3000)

    # GPU params
    parser.add_argument('--gpu_id',             type=int,       default=0)
    parser.add_argument('--data_parallel',      type=str,       default='True')

    # Misc params
    parser.add_argument('--num_workers',        type=int,       default=1)

    return parser


def main():
    # Args
    parser = config()
    args = parser.parse_args()

    # Environment
    set_random_seed(args.seed)
    device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu_id}')

    # Log
    log_dir = os.path.join(args.expt_dir, args.expt_name, args.run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)  # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/

    # Task
    task = 'cls' if args.dataset == 'multi_nli' else 'seq2seq'

    if args.split == 'train':
        # Load dataset
        train_dataset = BaseDataset(args.dataset, args.model, args.split, 'train')
        valid_dataset = BaseDataset(args.dataset, args.model, args.split, 'valid')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True, drop_last=True)

        # Get tokenizer
        tokenizer = train_dataset.get_tokenizer()

        # Load model
        model = get_model(args.model, task)

        # Set optimizer
        optimizer = optim.Adam(model.parameters(), args.lr)

        # Resume from checkpoint
        epoch = 1
        best_metric = -9999
        if args.ckpt:
            ckpt = torch.load(args.ckpt)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            epoch = ckpt['epoch']
            best_metric = ckpt['best_metric']

        # Move the model to the device
        model = model.to(device)

        # Train
        train(model=model,
              tokenizer=tokenizer,
              train_dataloader=train_loader,
              valid_dataloader=valid_loader,
              optimizer=optimizer,
              device=device,
              writer=writer,
              args={'best_metric': best_metric,
                    'epoch_s': epoch,
                    'epoch_e': args.epochs,
                    'accumulation_steps': args.acc_step,
                    'valid_interval': args.valid_interval,
                    'ckpt_interval': args.ckpt_interval,
                    'log_dir': log_dir,
                    'task_name': task}
              )


if __name__ == '__main__':
    main()
