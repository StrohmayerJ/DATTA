
import torch
import wandb
from utils.bns_utils import AlignFeatureStatsLossHook, choose_layers, collect_bn_params, compute_statistics, freeze_except_bn, set_random_weights
import torch.nn as nn
import numpy as np
import datasets as data
from os.path import exists
from os import listdir
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import random
import copy
from sklearn.model_selection import train_test_split
from utils.metrics import recallMetric, precisionMetric, create_confusion_matrix

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Enable deterministic behavior
S = 3407  # https://arxiv.org/abs/2109.08203
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
    metrics = {'acc': [],'precision': [],'recall': [],'f1': []}

    assert opt.momentum_mvg != 1.0 

    # get run names
    runDir = "runs"
    runBase = opt.name
    runs = [int(file[len(runBase) + 1:]) for file in listdir(runDir) if file.startswith(runBase + "_")] if exists(runDir) else ['']
    runs = sorted(runs)
    print("Run directory:", runDir)
    print("Run base name:", runBase)
    print("Runs: ", runs)

    # create Widar3g6 dataset
    print("Loading Widar3.0-G6D test dataset...")
    datasetSource = data.Widar3g6d(opt.data, augPath='', opt=opt,mode='TRAIN') 
    dataset = data.Widar3g6d(opt.data, augPath='', opt=opt,mode='TEST') 
    test_idx, target_val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=True, random_state=42)
    if opt.eval:
        test_subset = Subset(dataset, test_idx)
    else:
        test_subset = Subset(dataset, target_val_idx)
    dataloaderTarget = DataLoader(test_subset, batch_size=opt.bs, shuffle=False, num_workers=4)  


    print("Performing TTA...")
    for i in sorted(runs):

        # load DAT model checkpoint
        model_path = runDir+"/"+opt.name + (f"_{i}" if i != '' else '') + "/modelBestValLoss.pth"
        model = torch.load(model_path, map_location=device)
        model.to(device)

        # create source dataloader
        dataloaderSource = DataLoader(datasetSource, batch_size=128, shuffle=False, num_workers=opt.workers)
        
        # get source statistics
        print("Computing statistics...")
        list_stat_mean, list_stat_var = compute_statistics(model, dataloaderSource, opt)
        #del datasetSource, dataloaderSource

        confusion_matrix = torch.zeros(opt.classes, opt.classes)
        batch_count = 0
        model_origin = copy.deepcopy(model)

        for batch_id, batch in tqdm(enumerate(dataloaderTarget), total=len(dataloaderTarget)):

                setup_model_optimizer = False
                if batch_id == 0:
                        print(f'Initialize the model for online TTA')
                        setup_model_optimizer = True #  setup model and optimizer only before the first sample comes

                if setup_model_optimizer:
                    model = copy.deepcopy(model_origin)

                    # when we initialize the model, we have to re-choose the layers from it.
                    candidate_layers = [nn.LayerNorm]
                    chosen_layers = choose_layers(model, candidate_layers)
                    assert len(list_stat_mean) == len(chosen_layers)
                    assert len(list_stat_var) == len(chosen_layers)

                    # setup model and optimizer for training in online fashion
                    if opt.update_affine_only:
                            model = freeze_except_bn(model, bn_condidiate_layers=candidate_layers)  # set only batchnorm layers to trainable
                            params, _ = collect_bn_params(model,bn_candidate_layers=candidate_layers)  # collecting gamma and beta in batchnorm layers
                            optimizer = torch.optim.Adam(params, lr=opt.lr)
                    else:
                            optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.lr)

                    # setup loss, specifying which target statistic should be used for regularization, source or batch norm
                    if opt.stat_reg == 'source':
                                stat_reg_hooks = []
                                for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                                    for block_name in opt.chosen_blocks:
                                        if block_name in chosen_layer_name:
                                            stat_reg_hooks.append(
                                                # load statistcs, initialize the average meter for loss
                                                AlignFeatureStatsLossHook(chosen_layer, stats=(list_stat_mean[layer_id],list_stat_var[layer_id]), opt=opt))
                                            break

                model.eval()  # BN layers and DROPOUT are set to eval mode
                feature_window, c,d,_,_ = [x.to(device, non_blocking=True) for x in batch] 
                feature_window,d = feature_window.float(),d.long()
          
                loss_consis = torch.tensor(0).float().to(device)
                pred = model(feature_window)

                loss_alignment = torch.tensor(0).float().to(device)
                if opt.stat_reg:
                    for hook in stat_reg_hooks:
                        loss_alignment += hook.r_feature.to(device)

                loss = loss_consis + loss_alignment 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # remove hooks for inference
                if opt.stat_reg:
                    for stat_reg_hook in stat_reg_hooks:
                        stat_reg_hook.close()

                with torch.no_grad():
                    pred = model(feature_window)
                    batch_count += 1
                    confusion_matrix += create_confusion_matrix(pred['activity_recognizer_logits'], c, opt.classes)
 
                hook_layer_counter = 0
                for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                        for block_name in opt.chosen_blocks:
                            if block_name in chosen_layer_name:
                                stat_reg_hooks[hook_layer_counter].add_hook_back(chosen_layer)
                                hook_layer_counter += 1
                assert hook_layer_counter == len(stat_reg_hooks)

                # random weight reset
                set_random_weights(model, model_origin, percentage=opt.reset_probability)


        avg_acc = confusion_matrix.diag().sum()/confusion_matrix.sum()
        avg_recall = recallMetric(confusion_matrix)
        avg_precision = precisionMetric(confusion_matrix)
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        metrics['acc'].append(avg_acc)
        metrics['precision'].append(avg_precision)
        metrics['recall'].append(avg_recall)
        metrics['f1'].append(avg_f1)
        print(f"Run: {i} | P {avg_precision*100:.2f} | R {avg_recall*100:.2f} | F1 {avg_f1*100:.2f} | ACC {avg_acc*100:.2f}")
        wandb.log({"run": i, "Test/precision": avg_precision * 100, "Test/recall": avg_recall * 100, "Test/f1": avg_f1 * 100, "Test/acc": avg_acc * 100})

    print("Mean±Std ------------------------------------")
    metrics = {k: np.array(v) for k, v in metrics.items()}      
    meanPrecision = np.mean(np.array(metrics['precision']))
    stdPrecision = np.std(np.array(metrics['precision']))
    meanRecall = np.mean(np.array(metrics['recall']))
    stdRecall = np.std(np.array(metrics['recall']))
    meanF1 = np.mean(np.array(metrics['f1']))
    stdF1 = np.std(np.array(metrics['f1']))
    meanAcc = np.mean(np.array(metrics['acc']))
    stdAcc = np.std(np.array(metrics['acc']))
    wandb.log({"avg/mean/precision": meanPrecision * 100, "avg/std/precision": stdPrecision * 100, 
                   "avg/mean/recall": meanRecall * 100, "avg/std/recall": stdRecall * 100,
                   "avg/mean/f1": meanF1 * 100, "avg/std/f1": stdF1 * 100,
                   "avg/mean/acc": meanAcc * 100, "avg/std/acc": stdAcc * 100})
    print(f"Mean±Std | P {meanPrecision*100:.2f}±{stdPrecision*100:.2f} | R {meanRecall*100:.2f}±{stdRecall*100:.2f} | F1 {meanF1*100:.2f}±{stdF1*100:.2f} | ACC {meanAcc*100:.2f}±{stdAcc*100:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/widar3g6d/', help='data directory')
    parser.add_argument('--name', default='wdat', help='run base name of the DAT model')
    parser.add_argument('--bs', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--ws', type=int, default=220, help='spectrogram window size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--classes', type=int, default=6, help='number of classes in training dataset')
    parser.add_argument('--log', action='store_true', help='enable wandb logging')
    # TTA-specific arguments
    parser.add_argument('--exp_name', default='wdat_tta', help='TTA experiment wandb name')
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--stat_type', type=str, default='spatiotemp', help='type of aggregation, either spatiotemp or spatial')
    parser.add_argument('--update_affine_only', action='store_true', help='only update affine parameters')
    parser.add_argument('--chosen_blocks', type=str, nargs='+', default=['layers.0']) # can be either ['layers.0','layers.1','layers.2','layers.3'] 
    parser.add_argument('--stat_reg', type=str, default='source', help='either source or batch norm statistics bns')
    parser.add_argument('--reg_type', type=str, default='l2', help='either l1 or l2 or kld or jsd')
    parser.add_argument('--momentum_mvg', type=float, default=0.55)
    parser.add_argument('--moving_avg', type=bool, default=True)
    parser.add_argument('--noise_level', type=float, default=0.0155, help='noise level')
    parser.add_argument('--reset_probability', type=float, default=0.0001, help='probability of resetting weights')
    parser.add_argument('--eval', action='store_true', default=False)
    opt = parser.parse_args()

    # enable/disable wandb
    if opt.log:
        wandb.init(project=f"DATTA",entity="XXXX",name=opt.exp_name)
    else:
        wandb.init(mode="disabled")

    test(opt)
    torch.cuda.empty_cache()
