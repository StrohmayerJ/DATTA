
from models.wiflexformer_dat import WiFlexFormerDAT
import torch
import torch.nn as nn
import numpy as np
import sys
import time
import datasets as data
import os.path
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.metrics import recallMetric, precisionMetric, create_confusion_matrix

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import wandb
os.environ["WANDB_SILENT"] = "true"

def train(opt, runDir):
    # Select computing device
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    # Create Widar3g6d dataset
    print("Loading Widar3.0-G6D dataset...")
    dataset = data.Widar3g6d(opt.data, augPath=opt.augment, opt=opt, mode='TRAIN') 
    frac_for_valid = 0.2  # 80% train, 20% val
    sample_count = len(dataset)
    val_sample_count = int(sample_count * frac_for_valid)
    train_sample_count = sample_count - val_sample_count
    generator = torch.Generator()
    generator.manual_seed(42)
    datasetTrain, datasetVal = torch.utils.data.random_split(dataset, [train_sample_count, val_sample_count],generator=generator)

    # Training and validation dataloader
    dataloaderTrain = DataLoader(datasetTrain, batch_size=opt.bs, num_workers=opt.workers, drop_last=True, shuffle=True)
    dataloaderVal = DataLoader(datasetVal, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

    # init loss and metrics
    bestLossVal = sys.maxsize
    bestF1Val = 0

    # create WiFlexFormerDAT model
    model = WiFlexFormerDAT(grl_lambda=opt.ld)
    print("Model: " + str(model.__class__.__name__) + f" | #Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.to(device)

    # setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.001, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(dataloaderTrain), T_mult=1, eta_min=opt.lr / 10)

    # Training loop
    print("Training...")
    for epoch in tqdm(range(opt.epochs), desc='Epochs', unit='epoch'):
        start_time = time.time()
        model.train()
        epoch_loss_train = 0
        epoch_loss_train_supervised_activity = 0
        epoch_loss_train_unsupervised_activity = 0
        epoch_loss_train_supervised_domain = 0
        epoch_loss_train_smoothing = 0
        epoch_loss_train_confidence = 0
        epoch_loss_train_balance = 0
        batch_count_train = 0
        confusion_matrix_train_activity = torch.zeros(opt.classes, opt.classes)
        confusion_matrix_train_domain = torch.zeros(opt.domains, opt.domains)

        for batch in tqdm(dataloaderTrain, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='batch', leave=False):
            
            # adjust GRL lambda
            model.grl_lambda = (2. / (1. + np.exp(-10 * epoch / opt.epochs)) - 1)*opt.ld 
            
            # get training sample
            feature_window, c, d, _, _ = [x.to(device) for x in batch]  
            feature_window, d = feature_window.float(), d.long()

            # forward pass
            pred = model(feature_window)
            target = {'activity': c, 'domain': d}

            # Compute DAT loss
            loss_train = model.loss_dat(pred, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss_train.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Accumulate losses for logging
            epoch_loss_train += loss_train.item()
            epoch_loss_train_supervised_activity += model.loss_dat_supervised_activity(pred, target).item()
            epoch_loss_train_unsupervised_activity += model.loss_dat_unsupervised_activity(pred).item()
            epoch_loss_train_supervised_domain += model.loss_dat_supervised_domain(pred, target).item()
            epoch_loss_train_smoothing += model.loss_smoothing_constraint(pred['cls_token_output'], M=1).item()
            epoch_loss_train_confidence += model.loss_confidence_constraint(F.softmax(pred['activity_recognizer_logits'], dim=-1)).item()
            epoch_loss_train_balance += model.loss_balance_constraint(F.softmax(pred['activity_recognizer_logits'], dim=-1), target['domain']).item()

            # Compute training confusion matrices
            confusion_matrix_train_activity += create_confusion_matrix(pred['activity_recognizer_logits'], c, opt.classes)
            confusion_matrix_train_domain += create_confusion_matrix(pred['domain_discriminator_logits'], d, opt.domains)

            # Update batch count
            batch_count_train += 1

        # Normalize training metrics
        epoch_loss_train /= batch_count_train
        epoch_loss_train_supervised_activity /= batch_count_train
        epoch_loss_train_unsupervised_activity /= batch_count_train
        epoch_loss_train_supervised_domain /= batch_count_train
        epoch_loss_train_smoothing /= batch_count_train
        epoch_loss_train_confidence /= batch_count_train
        epoch_loss_train_balance /= batch_count_train
        epoch_precision_train_activity = precisionMetric(confusion_matrix_train_activity)
        epoch_recall_train_activity = recallMetric(confusion_matrix_train_activity)
        epoch_f1_train_activity = 2 * (epoch_precision_train_activity * epoch_recall_train_activity) / (epoch_precision_train_activity + epoch_recall_train_activity)
        epoch_precision_train_domain = precisionMetric(confusion_matrix_train_domain)
        epoch_recall_train_domain = recallMetric(confusion_matrix_train_domain)
        epoch_f1_train_domain = 2 * (epoch_precision_train_domain * epoch_recall_train_domain) / (epoch_precision_train_domain + epoch_recall_train_domain)

        # Validation
        epoch_loss_val = 0
        epoch_loss_val_supervised_activity = 0
        epoch_loss_val_unsupervised_activity = 0
        epoch_loss_val_supervised_domain = 0
        epoch_loss_val_smoothing = 0
        epoch_loss_val_confidence = 0
        epoch_loss_val_balance = 0
        batch_count_val = 0
        confusion_matrix_val_activity = torch.zeros(opt.classes, opt.classes)
        confusion_matrix_val_domain = torch.zeros(opt.domains, opt.domains)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloaderVal, desc='Validation', unit='batch', leave=False):
                feature_window, c, d, _, _ = [x.to(device) for x in batch]  # Widar3-G6_Domain
                feature_window, d = feature_window.float(), d.long()

                # Inference
                pred = model(feature_window)
                target = {'activity': c, 'domain': d}

                # Compute loss
                loss_val = model.loss_dat(pred, target)
                epoch_loss_val += loss_val.item()

                # Accumulate validation losses
                epoch_loss_val_supervised_activity += model.loss_dat_supervised_activity(pred, target).item()
                epoch_loss_val_unsupervised_activity += model.loss_dat_unsupervised_activity(pred).item()
                epoch_loss_val_supervised_domain += model.loss_dat_supervised_domain(pred, target).item()
                epoch_loss_val_smoothing += model.loss_smoothing_constraint(pred['cls_token_output'], M=1).item()
                epoch_loss_val_confidence += model.loss_confidence_constraint(F.softmax(pred['activity_recognizer_logits'], dim=-1)).item()
                epoch_loss_val_balance += model.loss_balance_constraint(F.softmax(pred['activity_recognizer_logits'], dim=-1), target['domain']).item()

                # Compute validation confusion matrices
                confusion_matrix_val_activity += create_confusion_matrix(pred['activity_recognizer_logits'], c, opt.classes)
                confusion_matrix_val_domain += create_confusion_matrix(pred['domain_discriminator_logits'], d, opt.domains)

                # Update batch count
                batch_count_val += 1

        # compute epoch losses and metrics
        epoch_loss_val /= batch_count_val
        epoch_loss_val_supervised_activity /= batch_count_val
        epoch_loss_val_unsupervised_activity /= batch_count_val
        epoch_loss_val_supervised_domain /= batch_count_val
        epoch_loss_val_smoothing /= batch_count_val
        epoch_loss_val_confidence /= batch_count_val
        epoch_loss_val_balance /= batch_count_val
        epoch_precision_val_activity = precisionMetric(confusion_matrix_val_activity)
        epoch_recall_val_activity = recallMetric(confusion_matrix_val_activity)
        epoch_f1_val_activity = 2 * (epoch_precision_val_activity * epoch_recall_val_activity) / (epoch_precision_val_activity + epoch_recall_val_activity)
        epoch_precision_val_domain = precisionMetric(confusion_matrix_val_domain)
        epoch_recall_val_domain = recallMetric(confusion_matrix_val_domain)
        epoch_f1_val_domain = 2 * (epoch_precision_val_domain * epoch_recall_val_domain) / (epoch_precision_val_domain + epoch_recall_val_domain)

        # Checkpointing based on validation performance
        if epoch_f1_val_activity > bestF1Val:
            bestF1Val = epoch_f1_val_activity
            print(f"Found better validation F1-Score, saving model...")
            torch.save(model, os.path.join(runDir, "modelBestValF1.pth"))
        if epoch_loss_val < bestLossVal:
            bestLossVal = epoch_loss_val
            print(f"Found better validation loss, saving model...")
            torch.save(model, os.path.join(runDir, "modelBestValLoss.pth"))

        # Logging local
        print(f"Epoch {epoch+1} [time {np.round(time.time() - start_time, 2)}s]")
        print(f"TRAIN Loss: {epoch_loss_train:.3f} | TRAIN S Activity Loss: {epoch_loss_train_supervised_activity:.3f} | TRAIN U Activity Loss: {epoch_loss_train_unsupervised_activity:.3f} | TRAIN S Domain Loss: {epoch_loss_train_supervised_domain:.3f} | TRAIN Smoothing Loss: {epoch_loss_train_smoothing:.3f} | TRAIN Confidence Loss: {epoch_loss_train_confidence:.3f} | TRAIN Balance Loss: {epoch_loss_train_balance:.3f} | TRAIN Activity F1-Score: {epoch_f1_train_activity:.3f} | TRAIN Domain F1-Score: {epoch_f1_train_domain:.3f}")
        print(f"VAL Loss: {epoch_loss_val:.3f} | VAL S Activity Loss: {epoch_loss_val_supervised_activity:.3f} | VAL U Activity Loss: {epoch_loss_val_unsupervised_activity:.3f} | VAL S Domain Loss: {epoch_loss_val_supervised_domain:.3f} | VAL Smoothing Loss: {epoch_loss_val_smoothing:.3f} | VAL Confidence Loss: {epoch_loss_val_confidence:.3f} | VAL Balance Loss: {epoch_loss_val_balance:.3f} | VAL Activity F1-Score: {epoch_f1_val_activity:.3f} | VAL Domain F1-Score: {epoch_f1_val_domain:.3f}")
        print("-------------------------------------------------------------------------")
        
        # Logging wandb
        wandb.log({
            'Other/lr': optimizer.param_groups[0]['lr'], 
            'Train/train Loss': epoch_loss_train, 
            'Train/train S Activity Loss': epoch_loss_train_supervised_activity,
            'Train/train U Activity Loss': epoch_loss_train_unsupervised_activity,
            'Train/train S Domain Loss': epoch_loss_train_supervised_domain,
            'Train/train Smoothing Loss': epoch_loss_train_smoothing,
            'Train/train Confidence Loss': epoch_loss_train_confidence,
            'Train/train Balance Loss': epoch_loss_train_balance,
            'Train/train Activity F1-Score': epoch_f1_train_activity,
            'Train/train Domain F1-Score': epoch_f1_train_domain,
            'Val/val Loss': epoch_loss_val, 
            'Val/val S Activity Loss': epoch_loss_val_supervised_activity,
            'Val/val U Activity Loss': epoch_loss_val_unsupervised_activity,
            'Val/val S Domain Loss': epoch_loss_val_supervised_domain,
            'Val/val Smoothing Loss': epoch_loss_val_smoothing,
            'Val/val Confidence Loss': epoch_loss_val_confidence,
            'Val/val Balance Loss': epoch_loss_val_balance,
            'Val/val Activity F1-Score': epoch_f1_val_activity, 
            'Val/val Domain F1-Score': epoch_f1_val_domain,
            'Other/GRL Lambda': model.grl_lambda
        }, step=epoch)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/widar3g6d/', help='data directory')
    parser.add_argument('--name', default='wadt', help='run base name (_* will be appended for multiple runs)')
    parser.add_argument('--epochs', type=int, default=4000, help='number of training epochs')
    parser.add_argument('--num', type=int, default=1, help='number of training runs')
    parser.add_argument('--lr', type=float, default=0.00005, help='optimizer learning rate')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--ws', type=int, default=220, help='spectrogram window size (number of WiFi packets)')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--augment', default='aug/default.yaml', type=str, metavar='PATH', help='path to augmentation parameters (default: aug/default.yaml)')
    parser.add_argument('--classes', type=int, default=6, help='number of classes in training dataset')
    parser.add_argument('--domains', type=int, default=7, help='number of domains in training dataset')
    parser.add_argument('--log', action='store_true', help='enable wandb logging')
    parser.add_argument('--ld', type=float, default=8, help='GRL lambda scaling factor')
    opt = parser.parse_args()

    # Conduct opt.num trining runs
    run_name = opt.name
    for i in range(opt.num):
        opt.name = run_name + f"_{i+1}"
        
        # enable/disable wandb
        if opt.log:
            wandb.init(project=f"DATTA",entity="XXXX",name=opt.exp_name)
        else:
            wandb.init(mode="disabled")
        
        # create run directory
        runDir = os.path.join("runs", opt.name)
        os.makedirs(runDir, exist_ok=True)

        # train WiFlexFormer DAT model
        train(opt, runDir)

        # Finish wandb run and clean up
        wandb.run.finish() if wandb.run else None
        torch.cuda.empty_cache()
