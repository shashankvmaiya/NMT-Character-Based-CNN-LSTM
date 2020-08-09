#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway
from utils import assert_expected_size

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size # word_embed_size
        self.char_embed_size = 50
        self.dropout_rate=0.3

        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=pad_token_idx)
        self.cnn = CNN(self.char_embed_size, self.embed_size)
        self.highway = Highway(self.embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """

        sentence_length, batch_size, m_word = input_tensor.size()
        e_char, e_word = self.char_embed_size, self.embed_size
        #print('input_tensor size = {}'.format(input_tensor.size()))

        # Reshaping input tensor with a revised batch_size = sentence_length*batch_size
        x_padded = torch.reshape(input_tensor, (sentence_length*batch_size, m_word))
        assert_expected_size(x_padded, 'x_padded', [sentence_length*batch_size, m_word])
        #print('x_padded size = {}'.format(x_padded.size()))

        x_emb = self.embeddings(x_padded)
        assert_expected_size(x_emb, 'x_emb', [sentence_length*batch_size, m_word, e_char])
        #print('x_emb size = {}'.format(x_emb.size()))

        x_reshaped = torch.reshape(x_emb, (sentence_length*batch_size, e_char, m_word))
        assert_expected_size(x_reshaped, 'x_reshaped', [sentence_length*batch_size, e_char, m_word])
        #print('x_reshaped size = {}'.format(x_reshaped.size()))

        x_conv_out = self.cnn.forward(x_reshaped)
        assert_expected_size(x_conv_out, 'x_conv_out', [sentence_length*batch_size, e_word])
        #print('x_conv_out size = {}'.format(x_conv_out.size()))

        x_highway = self.highway.forward(x_conv_out)
        assert_expected_size(x_highway, 'x_highway', [sentence_length*batch_size, e_word])
        #print('x_highway size = {}'.format(x_highway.size()))

        x_word_emb = self.dropout(x_highway)
        assert_expected_size(x_word_emb, 'x_word_emb', [sentence_length*batch_size, e_word])
        #print('x_word_emb size = {}'.format(x_word_emb.size()))

        output = torch.reshape(x_word_emb, (sentence_length, batch_size, e_word))
        assert_expected_size(output, 'output', [sentence_length, batch_size, e_word])
        #print('output size = {}'.format(output.size()))

        return output



