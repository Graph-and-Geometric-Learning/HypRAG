import torch
import hyplib.nn as hnn
from hyplib.optimizers.radam import RiemannianAdam
from hyplib.optimizers.rsgd import RiemannianSGD

from geoopt import ManifoldParameter
from torch.optim import AdamW, Optimizer
from transformers import get_scheduler, get_cosine_schedule_with_warmup


def get_param_groups(modules, args):
    decay = set()
    no_decay = set()
    blacklist_weight_modules = (torch.nn.LayerNorm, hnn.LorentzLayerNorm)
    named_parameters = [(name, param) for model in modules for name, param in model.named_parameters()]
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        # YUCK!!!
        if param.squeeze().ndim < 2:
            no_decay.add(name)
        elif "bias" in name:
            no_decay.add(name)
        elif isinstance(param, blacklist_weight_modules):
            no_decay.add(name)
        elif "logit_scale" in name:
            no_decay.add(name)
        else:
            decay.add(name)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in named_parameters if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": args.learning_rate},
    ]

    return optim_groups



class RAdamW(Optimizer):
    '''
    Taken from HypFormer
    '''

    def __init__(self, optim_groups, euc_optimizer_type='adam', euc_lr=0.01, euc_weight_decay=0.0, hyp_optimizer_type='radam', hyp_lr=0.01, 
                    hyp_weight_decay=0.0, stabilize=50, amsgrad=False, nesterov_euc=False, nesterov_hyp=False, momentum_euc=0.9, momentum_hyp=0.9):
        # Separate parameters for Euclidean and Hyperbolic parts of the model
        euc_groups, hyp_groups = [], []
        for group in optim_groups:
            euc_params, hyp_params = [], []
            for param in group['params']:
                if isinstance(param, ManifoldParameter):
                    hyp_params.append(param)
                else:
                    euc_params.append(param)
            if euc_params:
                euc_groups.append({'params': euc_params, 'weight_decay': group['weight_decay'], 'lr': group['lr']})
            if hyp_params:
                hyp_groups.append({'params': hyp_params, 'weight_decay': group['weight_decay'], 'lr': group['lr']})

        # Initialize Euclidean optimizer
        if euc_optimizer_type == 'adam':
            optimizer_euc = torch.optim.Adam(euc_groups, lr=euc_lr, weight_decay=euc_weight_decay, amsgrad=amsgrad)
        elif euc_optimizer_type == 'sgd':
            optimizer_euc = torch.optim.SGD(euc_groups, lr=euc_lr, weight_decay=euc_weight_decay, nesterov=nesterov_euc, momentum=momentum_euc)
        elif euc_optimizer_type == 'adamw':
            optimizer_euc = torch.optim.AdamW(euc_groups, lr=euc_lr, weight_decay=euc_weight_decay, amsgrad=amsgrad)
        else:
            raise NotImplementedError("Unsupported Euclidean optimizer type")

        # Initialize Hyperbolic optimizer if there are Hyperbolic parameters
        if len(hyp_groups) > 0:
            if hyp_optimizer_type == 'radam':
                optimizer_hyp = RiemannianAdam(hyp_groups, lr=hyp_lr, stabilize=stabilize, weight_decay=hyp_weight_decay, amsgrad=amsgrad)
            elif hyp_optimizer_type == 'rsgd':
                optimizer_hyp = RiemannianSGD(hyp_groups, lr=hyp_lr, stabilize=stabilize, weight_decay=hyp_weight_decay, nesterov=nesterov_hyp, momentum=momentum_hyp)
            else:
                raise NotImplementedError("Unsupported Hyperbolic optimizer type")

            # Store both optimizers
            self.optimizer = [optimizer_euc, optimizer_hyp]
        else:
            # Store only Euclidean optimizer if there are no Hyperbolic parameters
            self.optimizer = [optimizer_euc]

    def step(self):
        # Perform optimization step for each optimizer
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self, *args, **kwargs):
        # Reset gradients to zero for each optimizer
        for optimizer in self.optimizer:
            optimizer.zero_grad(*args, **kwargs)


class RAdamWScheduler(object):
    '''
    Taken from HypFormer
    '''

    def __init__(self, name, optimizers, num_warmup_steps, num_training_steps):
        self.lr_schedulers = [
            get_cosine_schedule_with_warmup(
                # name,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            for optimizer in optimizers.optimizer
        ]

    def step(self, *args, **kwargs):
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step(*args, **kwargs)

    def get_last_lr(self):
        return [lr_scheduler.get_last_lr() for lr_scheduler in self.lr_schedulers]


# adapted from https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1#diff-2075fa9c224b395be5bda85544dd36572b59c76c54562819eadadbf268602834R157s
# and using similar logic from openclip
def configure_optimizer(modules, args):
    
    optim_groups = get_param_groups(modules, args)

    if args.optimizer == "adamw":
        optimizer = AdamW(optim_groups, betas=(args.adam_beta1, args.adam_beta2), eps=args.eps)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(optim_groups, betas=(args.adam_beta1, args.adam_beta2), eps=args.eps)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(optim_groups, lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "adamw_and_radam":
        optimizer = RAdamW(
            optim_groups,
            euc_lr=args.learning_rate,
            euc_weight_decay=args.weight_decay,
            euc_optimizer_type="adamw",
            hyp_lr=args.learning_rate,
            hyp_weight_decay=args.weight_decay,
            hyp_optimizer_type="radam",
        )
    return optimizer


def configure_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    if not isinstance(optimizer, RAdamW):
        scheduler = get_cosine_schedule_with_warmup(
            # name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = RAdamWScheduler(
            name,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    return scheduler
