#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor
from torch.nn import Module, GRUCell, Linear, Dropout, Identity
import torch

__author__ = 'Nikita Kuzmin -- Lomonosov Moscow State University'
__docformat__ = 'reStructuredText'
__all__ = ['Decoder']


class Decoder(Module):

    def __init__(self,
                 annotation_dim: int,
                 output_dim: int,
                 nb_classes: int,
                 dropout_p: float,) \
            -> None:
        """Decoder with no attention.

        :param input_dim: Input features in the decoder.
        :type input_dim: int
        :param output_dim: Output features of the RNN.
        :type output_dim: int
        :param nb_classes: Number of output classes.
        :type nb_classes: int
        :param dropout_p: RNN dropout.
        :type dropout_p: float
        """
        super().__init__()

        self.dropout: Module = Dropout(p=dropout_p)

        self.rnn: Module = GRUCell(
            input_size=annotation_dim,
            hidden_size=output_dim)

        self.alignment = Linear(output_dim + annotation_dim, 1)
        self.projection = Linear(annotation_dim, output_dim)

        self.classifier: Module = Linear(
            in_features=output_dim,
            out_features=nb_classes)

    def forward(self,
                x: Tensor,
                n_steps: int,
                y=None) \
            -> Tensor:
        """
        Forward pass of the decoder.

        :param x: annotations.
        :type x: torch.Tensor
        :param n_steps: output seq len.
        :type n_steps: int
        :return: Output predictions.
        :rtype: torch.Tensor
        """

        # x.shape: (batch_size, seq_len, annotation_dim)
        x = x.transpose(0, 1)  # (seq_len, batch_size, annotation_dim)
        h = x[-1]
        h = self.projection(h)
        h_states = []

        # GRUCell has no attribute 'flatten_parameters'
        # self.rnn.flatten_parameters()
        x = self.dropout(x)

        for _ in range(n_steps):
            a_inp = h.unsqueeze(0).expand(x.shape[0], -1, -1)
            a_inp = torch.cat([x, a_inp], dim=-1)
            alignment = self.alignment(a_inp)  # (seq_len, batch, 1)
            weights = torch.softmax(alignment, dim=0)
            context = (weights * x).sum(0)
            h = self.rnn(context, h)
            h_states.append(h)

        h_states = torch.stack(h_states, dim=0)  # (out_seq_len, batch, h_size)
        h_states = h_states.transpose(0, 1)

        logits = self.classifier(h_states)
        return logits


        # h = self.rnn(self.dropout(x))[0]
        # return self.classifier(h)


def test():
    device = torch.device('cuda')
    m = Decoder(2 * 3, 5, 7, 0.1).to(device)
    x = torch.rand(11, 13, 2 * 3, device=device)
    y = m(x, 3)
    print(x.shape, y.shape)


if __name__ == '__main__':
    test()


# EOF
