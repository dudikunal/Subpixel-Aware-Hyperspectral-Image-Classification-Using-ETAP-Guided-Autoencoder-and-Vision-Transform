#demo.py
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from model import DSNet_ViT
from utils import AvgrageMeter, accuracy, output_metric, NonZeroClipper, print_args
from dataset import prepare_dataset
import numpy as np
import time
import os

# Argument parser
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--fix_random', action='store_true', default=True, help='fix randomness')
parser.add_argument('--gpu_id', default='0', help='GPU id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Salinas'], default='Salinas', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--model_name', choices=['conv2d_unmix', 'vit_unmix'], default='vit_unmix', help='DSNet')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
args = parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)

        optimizer.zero_grad()
        if 'unmix' in args.model_name:
            re_unmix_nonlinear, re_unmix, output = model(batch_data)

            band = re_unmix.shape[1] // 2
            output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
            re_unmix = re_unmix_nonlinear + output_linear

            sad_loss = torch.mean(torch.acos(torch.sum(batch_data * re_unmix, dim=1) /
                                             (torch.norm(re_unmix, dim=1, p=2) * torch.norm(batch_data, dim=1, p=2))))
            loss = criterion(output, batch_target) + sad_loss
        else:
            output = model(batch_data)
            loss = criterion(output, batch_target)

        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(output, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.cpu().numpy())
        pre = np.append(pre, p.cpu().numpy())

    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader, criterion, device):
    model.eval()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)

            if 'unmix' in args.model_name:
                re_unmix_nonlinear, re_unmix, output = model(batch_data)

                band = re_unmix.shape[1] // 2
                output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
                re_unmix = re_unmix_nonlinear + output_linear

                sad_loss = torch.mean(torch.acos(torch.sum(batch_data * re_unmix, dim=1) /
                                                 (torch.norm(re_unmix, dim=1, p=2) * torch.norm(batch_data, dim=1, p=2))))
                loss = criterion(output, batch_target) + sad_loss
            else:
                output = model(batch_data)
                loss = criterion(output, batch_target)

            prec1, t, p = accuracy(output, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.item(), n)
            top1.update(prec1[0].item(), n)
            tar = np.append(tar, t.cpu().numpy())
            pre = np.append(pre, p.cpu().numpy())

    return top1.avg, objs.avg, tar, pre

def main():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.fix_random:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    os.makedirs('./results/', exist_ok=True)
    label_train_loader, label_test_loader, band, height, width, num_classes, label, total_pos_true = prepare_dataset(args)

    # Model Creation
    if args.model_name == 'conv2d_unmix':
        model = DSNet(band, num_classes, args.patches, args.model_name)
    elif args.model_name == 'vit_unmix':
        model = DSNet_ViT(band, num_classes, args.patches)
    else:
        raise KeyError("{} model is unknown.".format(args.model_name))
    model = model.to(device)
    print("Model Name: {}".format(args.model_name))

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)
    # Optimizer
    apply_nonegative = NonZeroClipper()
    if 'unmix' in args.model_name:
        params = map(id, model.unmix_decoder.parameters())
        ignored_params = list(set(params))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params}, {'params': model.unmix_decoder.parameters(), 'lr': 3e-4}],
                                     lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

    if args.flag_test == 'test':
        model.eval()
        model_files = [f for f in os.listdir('./results/') if f.endswith('.pkl')]
        if model_files:
            model_path = os.path.join('./results/', sorted(model_files)[-1])
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError("No saved model found in './results/' directory.")

        val_acc, val_obj, tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, device)
        OA_val, AA_mean_val, Kappa_val, AA_val = output_metric(tar_v, pre_v)
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA_val, AA_mean_val, Kappa_val))
        print(AA_val)
    else:
        print("Start training")
        tic = time.time()
        min_val_obj, best_OA = 0.5, 0
        for epoch in range(args.epoches):
            scheduler.step()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer, device)
            OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
            print(f"Epoch: {epoch + 1:03d} | Train Acc: {train_acc:.4f} | Loss: {train_obj:.4f}")

            if 'unmix' in args.model_name:
                model.unmix_decoder.apply(apply_nonegative)

            if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
                model.eval()
                val_acc, val_obj, tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, device)
                OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
                print(f"OA: {OA2:.4f} | AA: {AA_mean2:.4f} | Kappa: {Kappa2:.4f}")
                print("*************************")

                if OA2 > min_val_obj and epoch > 10:
                    model_save_path = os.path.join('./results/', f"{args.dataset}_{args.model_name}_epoch{epoch}.pkl")
                    torch.save(model.state_dict(), model_save_path)
                    min_val_obj = OA2
                    best_epoch = epoch
                    best_OA = OA2
                    best_AA = AA_mean2
                    best_Kappa = Kappa2
                    best_each_AA = AA2

        toc = time.time()
        print("Running Time: {:.2f}".format(toc - tic))
        print("**************************************************")
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(best_OA, best_AA, best_Kappa))
        print(best_each_AA)
        print("**************************************************")
        print(f"Best Epoch: {best_epoch:03d} | Best OA: {best_OA:.4f}")
        print_args(vars(args))

if __name__ == '__main__':
    main()
