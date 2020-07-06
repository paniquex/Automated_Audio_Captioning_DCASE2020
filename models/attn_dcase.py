#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module
from torch import Tensor

from modules import Encoder
from modules import AttnDecoder as Decoder

__author__ = 'Nikita Kuzmin -- Lomonosov Moscow State University'
__docformat__ = 'reStructuredText'
__all__ = ['AttnDCASE']


class AttnDCASE(Module):

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
            annotation_dim=output_dim_encoder * 2,
            output_dim=output_dim_h_decoder,
            nb_classes=nb_classes,
            dropout_p=dropout_p_decoder)

    def forward(self,
                x: Tensor,
                y=None) \
            -> Tensor:
        """Forward pass of the baseline method.

        :param x: Input features.
        :type x: torch.Tensor
        :return: Predicted values.
        :rtype: torch.Tensor
        """
        annotation: Tensor = self.encoder(x)
        logits: Tensor = self.decoder(annotation, self.max_out_t_steps, y)
        return logits


def test():
    import torch
    device = torch.device('cuda')
    max_t = 17
    inp_dim = 3
    nb_classes = 13
    model = AttnDCASE(inp_dim, 5, 7, 0.1, 11, nb_classes, 0.2, max_t).to(device)
    x = torch.rand(19, 23, inp_dim, device=device)
    logits = model(x)
    print(x.shape, logits.shape)


if __name__ == '__main__':
    test()


# EOF
