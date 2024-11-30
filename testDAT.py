import torch
import numpy as np
import datasets as data
from os.path import exists
from os import listdir
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import random
from sklearn.model_selection import train_test_split
from utils.metrics import recallMetric, precisionMetric, create_confusion_matrix

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Enable deterministic behavior
S = 3407  # https://arxiv.org/abs/2109.08203 :)
random.seed(S)
np.random.seed(S)
torch.manual_seed(S)
torch.cuda.manual_seed(S)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Main test loop
def test(opt):
    
    # select computing device
    device = f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu"

    # initialize metrics
    metrics = {'rmse': [],'acc': [],'precision': [],'recall': [],'f1': []}

    # get run names
    runDir = "runs"
    runBase = opt.name
    runs = [int(file[len(runBase) + 1:]) for file in listdir(runDir) if file.startswith(runBase + "_")] if exists(runDir) else ['']
    runs = sorted(runs)
    print("Run directory:", runDir)
    print("Run base name:", runBase)
    print("Runs: ", runs)

    # load test dataset
    print("Loading Widar3.0-G6D test dataset...")
    dataset = data.Widar3g6d(opt.data, augPath='', opt=opt,mode='TEST') 
    test_idx, target_val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=True, random_state=42) # note: 10 percent reserved for TTA validation
    test_subset = Subset(dataset, test_idx)
    dataloaderTest = DataLoader(test_subset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

    # iterate over runs
    for i in sorted(runs):
        
        # load DAT model checkpoint
        model_path = runDir+"/"+opt.name + (f"_{i}" if i != '' else '') + "/modelBestValLoss.pth"
        model = torch.load(model_path, map_location=device)
        model.to(device)

        # initialize confusion matrix
        confusion_matrix = torch.zeros(opt.classes, opt.classes)
        batch_count = 0
        
        # test loop
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloaderTest):
                feature_window,c,d,_,_ = [x.to(device) for x in batch] 
                feature_window,d = feature_window.float(),d.long()
            
                # inference
                pred = model(feature_window)
                # accumulate confusion matrix
                confusion_matrix += create_confusion_matrix(pred['activity_recognizer_logits'], c, opt.classes)
                batch_count += 1

        # compute per-run metrics
        run_acc = confusion_matrix.diag().sum()/confusion_matrix.sum()
        run_recall = recallMetric(confusion_matrix)
        run_precision = precisionMetric(confusion_matrix)
        run_f1 = 2 * (run_precision * run_recall) / (run_precision + run_recall)
        metrics['acc'].append(run_acc)
        metrics['precision'].append(run_precision)
        metrics['recall'].append(run_recall)
        metrics['f1'].append(run_f1)
        print(f"Run: {i} | P {run_precision*100:.2f} | R {run_recall*100:.2f} | F1 {run_f1*100:.2f} | ACC {run_acc*100:.2f}")

    # compute cross-run metrics
    print("Mean±Std ------------------------------------")
    metrics = {k: np.array(v) for k, v in metrics.items()}
    # compute mean and std of metrics       
    meanPrecision = np.mean(np.array(metrics['precision']))
    stdPrecision = np.std(np.array(metrics['precision']))
    meanRecall = np.mean(np.array(metrics['recall']))
    stdRecall = np.std(np.array(metrics['recall']))
    meanF1 = np.mean(np.array(metrics['f1']))
    stdF1 = np.std(np.array(metrics['f1']))
    meanAcc = np.mean(np.array(metrics['acc']))
    stdAcc = np.std(np.array(metrics['acc']))
    # print mean+-std
    print(f"Mean±Std | P {meanPrecision*100:.2f}±{stdPrecision*100:.2f} | R {meanRecall*100:.2f}±{stdRecall*100:.2f} | F1 {meanF1*100:.2f}±{stdF1*100:.2f} | ACC {meanAcc*100:.2f}±{stdAcc*100:.2f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/widar3g6d/', help='data directory')
    parser.add_argument('--name', default='wdat', help='run base name (_* will be appended for multiple runs)')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--ws', type=int, default=220, help='spectrogram window size (number of WiFi packets)')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--classes', type=int, default=6, help='number of classes in training dataset')
    parser.add_argument('--domains', type=int, default=7, help='number of domains in training dataset')
    opt = parser.parse_args()

    test(opt)
    torch.cuda.empty_cache()
