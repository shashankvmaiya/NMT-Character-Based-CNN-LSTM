#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.utils

from utils import assert_expected_size

class Highway(nn.Module):
    def __init__(self, word_embed_size):
        '''Initializing Highway Model
        @param embed_size: word embed length
        '''
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.w_projection = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.w_gate = nn.Linear(word_embed_size, word_embed_size, bias=True)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        '''
        @param x_conv_out (Tensor): Tensor of padded source sentences with shape (b, e_word), where
                                    b = batch_size, e_word = word embedding length. These are the outputs
                                    from convolutional neural network
        @returns x_highway (Tensor) : Tensor of padded source sentences with shape (b, e_word)
        '''
        batch_size, e_word = len(x_conv_out), self.word_embed_size
        assert_expected_size(x_conv_out, 'x_conv_out', [batch_size, e_word])

        relu = nn.ReLU()
        x_proj = relu(self.w_projection(x_conv_out)) # shape = (b, e_word)
        assert_expected_size(x_proj, 'x_proj', [batch_size, e_word])

        x_gate = torch.sigmoid(self.w_gate(x_conv_out)) # shape = (b, e_word)
        assert_expected_size(x_gate, 'x_gate', [batch_size, e_word])

        x_highway = x_gate*x_proj + (1-x_gate)*x_conv_out
        assert_expected_size(x_highway, 'x_highway', [batch_size, e_word])
        return x_highway

