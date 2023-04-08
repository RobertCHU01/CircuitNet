# Copyright 2022 CircuitNet. All rights reserved.

import models
import torch

def build_model(opt):
    model = models.__dict__[opt.pop('model_type')](**opt)
    model.init_weights(**opt)
    if opt['test_mode']:
        model.eval()
    elif not opt['test_mode'] and opt['pretrained'] is not None:
        if opt['cpu']:
            model.load_state_dict(torch.load(opt['pretrained'], map_location=lambda storage, loc: storage)['state_dict'])
        else:
            model.load_state_dict(torch.load(opt['pretrained'])['state_dict'])
    return model
