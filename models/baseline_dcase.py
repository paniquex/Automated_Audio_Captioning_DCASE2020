#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module
from torch import Tensor
import torch

from modules import Encoder
from modules import Decoder

__author__ = 'Konstantinos Drossos, Nikita Kuzmin -- Lomonosov Moscow State University'
__docformat__ = 'reStructuredText'
__all__ = ['BaselineDCASE']


class BaselineDCASE(Module):

    def __init__(self,
                 input_dim_encoder: int,
                 hidden_dim_encoder: int,
                 output_dim_encoder: int,
                 dropout_p_encoder: float,
                 output_dim_h_decoder: int,
                 nb_classes: int,
                 dropout_p_decoder: float,
                 max_out_t_steps: int) \
            -> None:
        """Baseline method for audio captioning with Clotho dataset.

        :param input_dim_encoder: Input dimensionality of the encoder.
        :type input_dim_encoder: int
        :param hidden_dim_encoder: Hidden dimensionality of the encoder.
        :type hidden_dim_encoder: int
        :param output_dim_encoder: Output dimensionality of the encoder.
        :type output_dim_encoder: int
        :param dropout_p_encoder: Encoder RNN dropout.
        :type dropout_p_encoder: float
        :param output_dim_h_decoder: Hidden output dimensionality of the decoder.
        :type output_dim_h_decoder: int
        :param nb_classes: Amount of output classes.
        :type nb_classes: int
        :param dropout_p_decoder: Decoder RNN dropout.
        :type dropout_p_decoder: float
        :param max_out_t_steps: Maximum output time-steps of the decoder.
        :type max_out_t_steps: int
        """
        super().__init__()

        self.max_out_t_steps: int = max_out_t_steps

        self.encoder: Module = Encoder(
            input_dim=input_dim_encoder,
            hidden_dim=hidden_dim_encoder,
            output_dim=output_dim_encoder,
            dropout_p=dropout_p_encoder)

        self.decoder: Module = Decoder(
            input_dim=output_dim_encoder * 2,
            output_dim=output_dim_h_decoder,
            nb_classes=nb_classes,
            dropout_p=dropout_p_decoder)
        self.fc = torch.nn.Linear(512, 20480)
        #self.fc = torch.nn.Linear(in_features=512 * 2568,
        #                          out_features=self.max_out_t_steps * 2568)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the baseline method.

        :param x: Input features.
        :type x: torch.Tensor
        :return: Predicted values.
        :rtype: torch.Tensor
        """

        h_encoder: Tensor = self.encoder(x).mean(1)#.unsqueeze(1).expand(
           # -1, self.max_out_t_steps, -1)
        #h_encoder = h_encoder.mean(1)#.unsqueeze(1).expand(
            #-1, self.max_out_t_steps, -1)

        batch_size = h_encoder.shape[0]
        feature_size = h_encoder.shape[1]
        #h_encoder = h_encoder.reshape(h_encoder.shape[0], -1)
        h_encoder = self.fc(h_encoder)
        h_encoder = h_encoder.reshape(batch_size, self.max_out_t_steps, feature_size)
        return self.decoder(h_encoder)


# EOF
