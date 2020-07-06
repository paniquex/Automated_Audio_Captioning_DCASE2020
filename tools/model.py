#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, MutableSequence, \
    Callable, Optional, List, Union, MutableMapping
from platform import processor
from pathlib import Path

from torch import cuda, zeros, Tensor, load as pt_load
from torch.nn import Module
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np

import gc

from apex import amp
import torchaudio

import tqdm


from models import BaselineDCASE
from models.attn_dcase import AttnDCASE

__author__ = 'Konstantinos Drossos -- Tampere University, Nikita Kuzmin -- Lomonosov Moscow State University'
__docformat__ = 'reStructuredText'
__all__ = ['get_device', 'get_model',
           'module_epoch_passing',
           'module_forward_passing']


def get_device(force_cpu: bool) \
        -> Tuple[str, str]:
    """Gets the available device.

    :param force_cpu: Force CPU usage?
    :type force_cpu: bool
    :return: Device and device name.
    :rtype: str, str
    """
    return ('cuda:0', cuda.get_device_name(cuda.current_device())) \
        if cuda.is_available() and not force_cpu else \
        ('cpu', processor())


def get_model(settings_model: MutableMapping[str, Union[str, MutableMapping]],
              settings_io: MutableMapping[str, Union[str, MutableMapping]],
              settings_training: MutableMapping[str, Union[str, MutableMapping]],
              output_classes: int,
              return_optimizer: bool=False,
              return_scheduler: bool=False,
              device: str='cpu') \
        -> (Module, Optimizer):
    """Creates and returns the model for the process.

    :param settings_model: Model specific settings to be used.
    :type settings_model: dict
    :param settings_io: File I/O settings to be used.
    :type settings_io: dict
    :param output_classes: Amount of output classes.
    :type output_classes: int
    :param return_optimizer: if True, then optimizer also will be returned
    :type return_optimizer: bool
    :return: Model.
    :rtype: torch.nn.Module, torch.optim.Optimizer
    """
    encoder_settings = settings_model['encoder']
    decoder_settings = settings_model['decoder']
    decoder_settings.update({'nb_classes': output_classes})

    kwargs = {**encoder_settings, **decoder_settings}
    if settings_training['use_attention']:
        model = AttnDCASE(**kwargs)
    else:
        model = BaselineDCASE(**kwargs)
    optimizer = Adam(params=model.parameters(),
                     lr=settings_training['optimizer']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    print("Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)
    if settings_training['use_apex']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=settings_training['apex_opt_level'])
    model = DataParallel(model)

    if settings_model['use_pre_trained_model']:
        model.load_state_dict(pt_load(Path(
            settings_io['root_dirs']['outputs'],
            settings_io['model']['model_dir'],
            settings_io['model']['pre_trained_model_name']
        ))['model'])
        optimizer.load_state_dict(pt_load(Path(
            settings_io['root_dirs']['outputs'],
            settings_io['model']['model_dir'],
            settings_io['model']['pre_trained_model_name']
        ))['optimizer'])

    if return_optimizer and return_scheduler:
        return model, optimizer, scheduler
    elif return_optimizer:
        return model, optimizer
    else:
        return model


def module_epoch_passing(data: DataLoader,
                         module: Module,
                         objective: Union[Callable[[Tensor, Tensor], Tensor], None],
                         optimizer: Union[Optimizer, None],
                         grad_norm: Optional[int] = 1,
                         grad_norm_val: Optional[float] = -1.,
                         settings_features = None,
                         settings_training = None,
                         use_apex=False) \
        -> Tuple[Tensor, List[Tensor], List[Tensor], List[str]]:
    """One full epoch passing.

    :param data: Data of the epoch.
    :type data: torch.utils.data.DataLoader
    :param module: Module to use.
    :type module: torch.nn.Module
    :param objective: Objective for the module.
    :type objective: callable|None
    :param optimizer: Optimizer for the module.
    :type optimizer: torch.optim.Optimizer | None
    :param grad_norm: Norm of the gradient for gradient clipping.
                      Defaults to 1. .
    :type grad_norm: int
    :param grad_norm_val: Max value for gradient clipping. If -1, then\
                          no clipping will happen. Defaults to -1. .
    :type grad_norm_val: float
    :return: Predicted and ground truth values\
             (if specified).
    :rtype: torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[str]
    """
    has_optimizer = optimizer is not None
    objective_output: Tensor = zeros(len(data))

    output_y_hat = []
    output_y = []
    f_names = []
    preprocessor = torchaudio.transforms.MelSpectrogram(sample_rate=settings_features['process']['sr'],
                                                        n_fft=settings_features['process']['nb_fft'],
                                                        hop_length=settings_features['process']['hop_size'],
                                                        f_min=settings_features['process']['f_min'],
                                                        f_max=settings_features['process']['f_max'],
                                                        n_mels=settings_features['process']['nb_mels'])
                                                        #power=settings_features['process']['power'])

    for i, example in enumerate(tqdm.tqdm(data)):
        with torch.no_grad():
            device = next(module.parameters()).device
            #preprocessor = torchaudio.transforms.MelSpectrogram(sample_rate=settings_features['process']['sr'],
            #                                                    n_fft=settings_features['process']['nb_fft'],
            #                                                    hop_length=settings_features['process']['hop_size'],
            #                                                    f_min=settings_features['process']['f_min'],
            #                                                    f_max=settings_features['process']['f_max'],
            #                                                    n_mels=settings_features['process']['nb_mels'])
            preprocessor = preprocessor.to(device)
            example[0] = torch.log(preprocessor.forward(example[0].to(device)) + np.finfo(float).eps)
            example[0] = torch.transpose(example[0], 1, 2)
            example[1] = example[1].to(device)
            #del device, preprocessor
            del device
            gc.collect()
        y_hat, y, f_names_tmp = module_forward_passing(example, module)
        f_names.extend(f_names_tmp)
        y = y[:, 1:]
        try:
            output_y_hat.extend(y_hat.cpu())
            output_y.extend(y.cpu())
        except AttributeError:
            pass
        except TypeError:
            pass
        except Exception:
            pass

        #try:
        #if settings_training['remove_duplicates_from_predicted_caption']:
        #    tmp = torch.argmax(y_hat, dim=2)

         #   tmp_shifted = torch.cat((tmp[:, 1:], torch.zeros((80, 1), dtype=int).cuda()), dim=1)
         #   mask = tmp != tmp_shifted
         #   print(mask.shape)
         #   print(mask)
         #   y_hat = npy_hat[mask]
         #   del tmp
         #   print(y_hat[0])
         #   print(y_hat[0].shape)
         #   exit()
        labels_length = min(y_hat.shape[1], y.size()[1])
        y_hat = y_hat[:, :labels_length, :]
        y = y[:, :labels_length]
        if settings_training:
            if settings_training['use_scst']:
                loss = objective(y_hat,
                          y, f_names_tmp)
                print(loss, 'AAAA')
            else:
                loss = objective(y_hat.contiguous().view(-1, y_hat.size()[-1]),
                         y.contiguous().view(-1))


        if has_optimizer:
            optimizer.zero_grad()
            if grad_norm_val > -1:
                clip_grad_norm_(module.parameters(),
                                max_norm=grad_norm_val,
                                norm_type=grad_norm)
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        if objective is not None:
            objective_output[i] = loss.cpu().item()
            del loss, labels_length
        gc.collect()
        #except TypeError:
        #    pass
        #except Exception:
        #    pass
        del y_hat, y, f_names_tmp
        gc.collect()
    return objective_output, output_y_hat, output_y, f_names


def module_forward_passing(data: MutableSequence[Tensor],
                           module: Module) \
        -> Tuple[Tensor, Tensor, List[str]]:
    """One forward passing of the module.

    Implements one forward passing of the module `module`, using the provided\
    data `data`. Returns the output of the module and the ground truth values.

    :param data: Input and output values for current forward passing.
    :type data: list[torch.Tensor]
    :param module: Module.
    :type module: torch.nn.Module
    :return: Output of the module and target values.
    :rtype: torch.Tensor, torch.Tensor, list[str]
    """
    #device = next(module.parameters()).device
    x, y, f_names = [i if isinstance(i, Tensor)
                     else i for i in data]
    return module(x), y, f_names

# EOF