import random
import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

# Modified source code from: https://github.com/wlin-at/ViTTA/blob/main/utils/BNS_utils.py

l1_loss = nn.L1Loss(reduction='mean')
l2_loss = nn.MSELoss(reduction='mean')

CANDIDATE_BN_LAYERS = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

def compute_kld(mean_true, mean_pred, var_true, var_pred):
    # mean1 and std1 are for true distribution
    # mean2 and std2 are for pred distribution
    # kld_mv = torch.log(std_pred / std_true) + (std_true ** 2 + (mean_true - mean_pred) ** 2) / (2 * std_pred ** 2) - 0.5

    kld_mv = 0.5 * torch.log(torch.div(var_pred, var_true)) + (var_true + (mean_true - mean_pred) ** 2) / \
             (2 * var_pred) - 0.5
    kld_mv = torch.sum(kld_mv)
    return kld_mv

class BNFeatureHook():
    def __init__(self, module, reg_type='l2norm', running_manner = False, use_src_stat_in_reg = True, momentum = 0.1):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.reg_type = reg_type
        self.running_manner = running_manner
        self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change
        if self.use_src_stat_in_reg:
            self.source_mean = module.running_mean.data
            self.source_var = module.running_var.data
        if self.running_manner:
            # initialize the statistics of computation in running manner
            self.mean = torch.zeros_like( module.running_mean)
            self.var = torch.zeros_like(module.running_var)
        self.momentum = momentum

    def hook_fn(self, module, input, output):  # input in shape (B, C, T, H, W)

        nch = input[0].shape[1]
        if isinstance(module, nn.BatchNorm1d):
            # input in shape (B, C) or (B, C, T)
            if len(input[0].shape) == 2: #  todo  BatchNorm1d in TAM G branch  input is (N*C,  T )
                batch_mean = input[0].mean([0,])
                batch_var = input[0].permute(1, 0,).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
            elif len(input[0].shape) == 3:  # todo BatchNorm1d in TAM L branch  input is (N, C, T)
                batch_mean = input[0].mean([0,2])
                batch_var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
        elif isinstance(module, nn.BatchNorm2d):
            # input in shape (B, C, H, W)
            batch_mean = input[0].mean([0, 2, 3])
            batch_var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
        elif isinstance(module, nn.BatchNorm3d):
            # input in shape (B, C, T, H, W)
            batch_mean = input[0].mean([0, 2, 3, 4])
            batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1,  unbiased=False)  # compute the variance along each channel

        self.mean =  self.momentum * batch_mean + (1.0 - self.momentum) * self.mean.detach() if self.running_manner else batch_mean
        self.var = self.momentum * batch_var + (1.0 - self.momentum) * self.var.detach() if self.running_manner else batch_var
        # todo if BN layer is set to eval, these two are the same;  otherwise, module.running_xx.data keeps changing
        self.mean_true = self.source_mean if self.use_src_stat_in_reg else module.running_mean.data
        self.var_true = self.source_var if self.use_src_stat_in_reg else module.running_var.data
        self.r_feature = compute_regularization(mean_true = self.mean_true, mean_pred = self.mean, var_true=self.var_true, var_pred = self.var, reg_type = self.reg_type)


    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn) 

    def close(self):
        self.hook.remove()

class TempStatsRegHook():
    def __init__(self, module, clip_len = None, temp_stats_clean_tuple = None, reg_type='l2norm', ):

        self.hook = module.register_forward_hook(self.hook_fn) 
        self.clip_len = clip_len

        self.reg_type = reg_type
        self.source_mean, self.source_var = temp_stats_clean_tuple

        self.source_mean = torch.tensor(self.source_mean).cuda()
        self.source_var = torch.tensor(self.source_var).cuda()

        self.mean_avgmeter = AverageMeterTensor()
        self.var_avgmeter = AverageMeterTensor()

        # self.momentum = momentum

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t

            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')
        # spatial_dim = h * w
        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        batch_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        batch_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # batch_mean = output.mean(2).mean((0, 2,3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # batch_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2,3)) # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


        self.mean_avgmeter.update(batch_mean, n= bz)
        self.var_avgmeter.update(batch_var, n= bz)

        if self.reg_type == 'l2norm':
            # # todo sum of squared difference,  averaged over  h * w
            # self.r_feature = torch.sum(( self.source_var - self.var_avgmeter.avg )**2 ) / spatial_dim + torch.sum(( self.source_mean - self.mean_avgmeter.avg )**2 ) / spatial_dim
            self.r_feature = torch.norm(self.source_var - self.var_avgmeter.avg, 2) + torch.norm(self.source_mean - self.mean_avgmeter.avg, 2)
        else:
            raise NotImplementedError

    def close(self):
        self.hook.remove()

class ComputeSpatioTemporalStatisticsHook():
    def __init__(self, module, clip_len = None,):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t
            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')

        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        self.temp_mean = output.mean((0, 2,3,4)).mean(0) #  (N, C, T, H, W)  ->   (C, )
        self.temp_var = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False) #  (N, C, T, H, W) -> (C, N, T, H, W) -> (C, )

        # batch_mean = input[0].mean([0, 2, 3, 4])
        # batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel

        self.temp_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # self.temp_mean = output.mean(2).mean((0, 2, 3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2, 3) )   # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


    def close(self):
        self.hook.remove()


class ComputeTemporalStatisticsHook():
    def __init__(self, module, clip_len = None,):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t
            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')

        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        self.temp_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # self.temp_mean = output.mean(2).mean((0, 2, 3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2, 3) )   # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


    def close(self):
        self.hook.remove()

def choose_layers(model, candidate_layers):
        chosen_layers = []
        counter = [0] * len(candidate_layers)
        for nm, m in model.named_modules():
            for candidate_idx, candidate in enumerate(candidate_layers):
                if isinstance(m, candidate):
                    counter[candidate_idx] += 1
                    chosen_layers.append((nm, m))
        return chosen_layers

def freeze_except_bn(model, bn_condidiate_layers, ):
    """
    freeze the model, except the BN layers
    :param model:
    :param bn_condidiate_layers:
    :return:
    """

    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        for candidate in bn_condidiate_layers:
            if isinstance(m, candidate):
                m.requires_grad_(True)
    return model

def collect_bn_params(model, bn_candidate_layers):
    params = []
    names = []
    for nm, m in model.named_modules():
        for candidate in bn_candidate_layers:
            if isinstance(m, candidate):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']: # weight is scale gamma, bias is shift beta
                        params.append(p)
                        names.append( f"{nm}.{np}")
    return params, names


class AlignFeatureStatsLossHook():
    """
    Combine regularization of several types of statistics
    todo if there are multiple views, compute the statistics on the volume of multiple views , and align statistics with the source statistics,  only one regularization
    """
    def __init__(self, module,
                 stats = None,
                 opt = None,
                 ):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.reg_type = opt.reg_type
        self.moving_avg = opt.moving_avg
        self.momentum = opt.momentum_mvg
        self.stat_type = opt.stat_type

        self.source_mean, self.source_var = stats

        self.device = torch.device(f"cuda:{opt.device}")
        if self.source_mean is not None and self.source_var is not None:
            self.source_mean, self.source_var = torch.tensor(self.source_mean).to(self.device), torch.tensor(self.source_var).to(self.device)

        if self.moving_avg:
                self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = MovingAverageTensor(momentum=self.momentum, device=self.device), MovingAverageTensor(momentum=self.momentum, device=self.device)
                #TODO in the paper they start from scratch
                self.mean_avgmeter_spatiotemp.avg = self.source_mean
                self.var_avgmeter_spatiotemp.avg = self.source_var
        else:
                self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = AverageMeterTensor(device=self.device), AverageMeterTensor(device=self.device)

    def hook_fn(self, module, input, output):
        self.r_feature = torch.tensor(0).float().to(self.device)
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d): #todo on BatchNorm2d and Batchnorm3d, all types of statistics
                raise NotImplementedError('Temporal statistics computation for nn.BatchNorm2d not implemented!')
                if isinstance(module, nn.BatchNorm2d):
                    # output is in shape (N*T,  C,  H,  W)
                    nt, c, h, w = output.size()
                    t = self.clip_len
                    bz = nt // t
                    output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    # output is in shape (N, C, T, H, W)
                    bz, c, t, h, w = output.size()
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz, c, t, h, w)

                self.compute_reg_for_NCTHW(output)
        elif isinstance(module, nn.LayerNorm):
            assert len(output.size()) == 3
            t, b, c = output.size()
            output = output.permute(1, 2, 0).contiguous() # wifi: t , b, c -> b, c, t # video: b, t, h, w, c ->  b, c, t, h, w (batch_first == True)
            self.compute_reg_for_NCTHW(output)


    def compute_reg_for_NCTHW(self, output):
        b, c, t = output.size()

        if 'spatiotemp' == self.stat_type:
            batch_mean_spatiotemp = output.mean((0, 2)) # (N, C, T) -> (C, )
            batch_var_spatiotemp = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased=False) # (N, C, T) -> (C, N, T) -> (C, )
            if self.moving_avg:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp)
            else:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp, n=b)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp, n=b)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean,
                                                                     self.mean_avgmeter_spatiotemp.avg,
                                                                     self.source_var,
                                                                     self.var_avgmeter_spatiotemp.avg, self.reg_type)
        elif 'spatial' == self.stat_type:
            raise NotImplementedError('Compute spatiotemp statistics not implemented!')

    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module

    def close(self):
        self.hook.remove()

class ComputeNormStatsHook():
    def __init__(self, module, stat_type, window_len = None, batch_size = None):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.window_len = window_len
        self.batch_size = batch_size
        self.stat_type = stat_type # spatiotemp or spatial

    def hook_fn(self, module, input, output):
            assert len(output.size()) == 3
            t, b, c = output.size()
            output = output.permute(1, 2, 0).contiguous() # wifi: t , b, c -> b, c, t # video: b, t, h, w, c ->  b, c, t, h, w (batch_first == True)
            self.compute_stat_for_NCTHW(output)

    def compute_stat_for_NCTHW(self, output):
        b, c, t = output.size()
        if self.stat_type == 'spatiotemp':
            self.batch_mean = output.mean((0, 2))  # (N, C, T) ->  (C, )
            self.batch_var = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T)  ->  (C, N, T) -> (C, )
        elif self.stat_type == 'spatial':
            self.batch_mean = output.mean((0))  # (N, C, T, H, W) ->  (C, T)
            self.batch_var = output.permute(1, 2, 0).contiguous().view([c, t, -1]).var(-1, unbiased=False)  # (N, C, T)  ->  (C, T, N) -> (C, T )

    def close(self):
        self.hook.remove()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MovingAverageTensor(object):
    def __init__(self, momentum=0.1, device=torch.device('cpu')):
        self.momentum = momentum
        self.device = device
        self.reset()
    def reset(self):
        self.avg = torch.tensor(0).float().to(self.device)
    def update(self, val):
        self.avg = self.momentum * val  + (1.0 - self.momentum) * self.avg.detach().to(val.device)

class AverageMeterTensor(object):
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.reset()
    def reset(self):
        self.val = torch.tensor(0).float().to(self.device)
        self.avg = torch.tensor(0).float().to(self.device)
        self.sum = torch.tensor(0).float().to(self.device)
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum.detach() + val * n
        self.count += n
        self.avg = self.sum / self.count
            
def set_random_weights(network_a, network_b, percentage=0.1):
    if percentage == 0:
        return
    with torch.no_grad():  # Disable gradient tracking for manual weight setting
        # Ensure both networks are on the same device
        device = next(network_a.parameters()).device
        network_b = network_b.to(device)
        
        # Iterate over each layer's parameters in both networks
        for param_a, param_b in zip(network_a.parameters(), network_b.parameters()):
            # Reshape the parameters to make indexing easier
            param_a_flat = param_a.reshape(-1)
            param_b_flat = param_b.reshape(-1)

            # compute random number between 0 and percentage
            #percentage = random.uniform(0, percentage)
            
            # Calculate the number of elements to replace
            num_elements = param_a_flat.size(0)
            
            # calculate random vector between 0 and 1
            random_vector = torch.rand(num_elements).to(device)
            # calculate mask for replacement
            mask = random_vector < percentage
            # Replace values at the selected indices
            param_a_flat[mask] = param_b_flat[mask].clone()

def compute_pred_consis(preds, n_views):
    """
    :param preds:  in shape (batch_size, n_views, n_class) before softmax
    :return:
    """
    bs, n_class = preds.size()
    bs //= n_views
    preds = preds.view([bs, n_views, n_class])
    softmaxs = []
    for view_id in range(n_views):
        softmaxs += [F.softmax( preds[:, view_id, :], dim=1)]

    # avg_softmax = torch.stack(softmaxs, dim=0).mean(0).detach()
    avg_softmax = torch.stack(softmaxs, dim=0).mean(0)

    loss_consis = [ l1_loss(softmaxs[view_id], avg_softmax)        for view_id in range(n_views) ]
    # loss_consis = [  kl_div(  preds[:, view_id, :]  , avg_softmax)     for view_id in range(n_views)   ]
    loss_consis = sum(loss_consis) / n_views
    return loss_consis

def compute_regularization(mean_true, mean_pred, var_true, var_pred, reg_type):
    mean_pred = mean_pred.to(mean_true.device)
    var_pred = var_pred.to(var_true.device)
    if reg_type == 'l1':
        return  l1_loss(mean_true, mean_pred) + l1_loss(var_true, var_pred)
    elif reg_type == 'l2':
        return l2_loss(mean_true, mean_pred) + l2_loss(var_true, var_pred) 
    elif reg_type == 'kld':
        return compute_kld(mean_true, mean_pred, var_true, var_pred)
    else:
        raise NotImplementedError(f'unknown regularization type {reg_type}')

def compute_statistics(model, data_loader, opt):
        device = f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu"
        
        compute_stat_hooks = []
        list_stat_mean = []
        list_stat_var = []
        candidate_layers = [nn.LayerNorm]
        chosen_layers = choose_layers(model, candidate_layers)

        for layer_id, (layer_name, layer_) in enumerate(chosen_layers):
            compute_stat_hooks.append(ComputeNormStatsHook(layer_, stat_type=opt.stat_type, window_len = opt.ws, batch_size=opt.bs))
            list_stat_mean.append(AverageMeter())
            list_stat_var.append(AverageMeter())

        model.eval()  
        with torch.no_grad():
            for batch in tqdm(data_loader):
                feature_window, _,_,_,_ = [x for x in batch] 
                feature_window = feature_window.float()
                feature_window = feature_window.to(device)
                _ = model(feature_window) 

                for hook_id, stat_hook in enumerate(compute_stat_hooks):
                    list_stat_mean[hook_id].update(stat_hook.batch_mean, n=opt.bs)
                    list_stat_var[hook_id].update(stat_hook.batch_var, n=opt.bs)

        for hook_id, stat_hook in enumerate(compute_stat_hooks):
            list_stat_mean[hook_id] = list_stat_mean[hook_id].avg.cpu().numpy()
            list_stat_var[hook_id] = list_stat_var[hook_id].avg.cpu().numpy()
        
        return list_stat_mean, list_stat_var

