#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.utils
from utils import assert_expected_size


class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        '''Initializing CNN Model
        @param char_embed_size: character embed length = e_char
        @param word_embed_size: word embed length = e_word
        @param kernel_size: kernel size
        '''
        self.kernel_size = kernel_size
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        super(CNN, self).__init__()
        # in_channels = e_char = char_embed_size
        # outchannels = e_word = word_embed_size
        self.cnn = nn.Conv1d(char_embed_size, word_embed_size, kernel_size, bias=True)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        '''
        @param x_reshaped (Tensor): Tensor of padded source sentences with shape (b, e_char, m_word), where
                                    b = batch_size, e_word = word embedding length and m_word = max_word_length = max characters in a word
        @returns x_conv_out (Tensor) : Tensor of padded source sentences with shape (b, e_word)
        '''
        batch_size, m_word = len(x_reshaped), len(x_reshaped[0][0])
        e_char, e_word, kernel_size = self.char_embed_size, self.word_embed_size, self.kernel_size

        #print('x_reshaped size = {}'.format(x_reshaped.size()))
        assert_expected_size(x_reshaped, 'x_reshaped', [batch_size, e_char, m_word])

        x_conv = self.cnn(x_reshaped)
        assert_expected_size(x_conv, 'x_conv', [batch_size, e_word, m_word-kernel_size+1])
        #print('x_conv size = {}'.format(x_conv.size()))

        relu = nn.ReLU()
        maxpool = nn.MaxPool1d(m_word-kernel_size+1)
        x_conv_out = relu(x_conv)
        #print('x_conv_out (after relu) size = {}'.format(x_conv_out.size()))
        x_conv_out = maxpool(x_conv_out)
        #print('x_conv_out (after maxpool) size = {}'.format(x_conv_out.size()))
        x_conv_out = torch.squeeze(x_conv_out, dim=2)
        #print('x_conv_out size (after squeeze) = {}'.format(x_conv_out.size()))
        assert_expected_size(x_conv_out, 'x_conv_out', [batch_size, e_word])

        return x_conv_out

