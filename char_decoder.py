#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from utils import assert_expected_size

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.vocab_size = len(target_vocab.char2id)
        self.char_embedding_size = char_embedding_size
        self.hidden_size = hidden_size
        pad_token_idx = target_vocab.char2id['<pad>']

        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, self.vocab_size, bias=True)
        self.decoderCharEmb = nn.Embedding(self.vocab_size, char_embedding_size, padding_idx=pad_token_idx)
        self.target_vocab = target_vocab

        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        length, batch_size = input.size() # length = m_word = number of characters in the word

        x_embeddings = self.decoderCharEmb(input)
        assert_expected_size(x_embeddings, 'x_embeddings', [length, batch_size, self.char_embedding_size])

        enc_hiddens, (hn, cn) = self.charDecoder(x_embeddings, dec_hidden)
        assert_expected_size(enc_hiddens, 'enc_hiddens', [length, batch_size, self.hidden_size])
        assert_expected_size(hn, 'hn', [1, batch_size, self.hidden_size])
        assert_expected_size(cn, 'cn', [1, batch_size, self.hidden_size])

        scores = self.char_output_projection(enc_hiddens)
        assert_expected_size(scores, 'scores', [length, batch_size, self.vocab_size])

        return scores, (hn, cn)
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        length, batch_size = char_sequence.size() # length = m_word = number of characters in the word

        scores, (hn, cn) = self.forward(char_sequence, dec_hidden)
        assert_expected_size(scores, 'scores', [length, batch_size, self.vocab_size])
        assert_expected_size(hn, 'hn', [1, batch_size, self.hidden_size])
        assert_expected_size(cn, 'cn', [1, batch_size, self.hidden_size])

        loss = nn.CrossEntropyLoss(reduction='sum')
        cross_entropy_loss = 0
        for i in range(batch_size):
            if self.target_vocab.end_of_word in char_sequence[:,i]:
                end_id = (char_sequence[:,i]==self.target_vocab.end_of_word).nonzero()
                cross_entropy_loss += loss(scores[:end_id,i,:], char_sequence[1:end_id+1,i])

        return cross_entropy_loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        batch_size = len(initialStates[0][0])
        current_char_id = torch.tensor([self.target_vocab.start_of_word]*batch_size).reshape((1, batch_size))
        assert_expected_size(current_char_id, 'current_char_id', [1, batch_size])

        decodedWords = ['']*batch_size
        dec_hidden = initialStates
        for i in range(max_length):
            scores, dec_hidden = self.forward(current_char_id, dec_hidden)
            assert_expected_size(scores, 'scores', [1, batch_size, self.vocab_size])
            current_char_id = torch.argmax(scores, dim=2)
            decodedWords = [d+self.target_vocab.id2char[c.item()] for (c, d) in zip(current_char_id.squeeze(dim=0), decodedWords)]

        decodedWords = [d[:d.find('}')] if '}' in d else d for d in decodedWords]
        return decodedWords

        ### END YOUR CODE

