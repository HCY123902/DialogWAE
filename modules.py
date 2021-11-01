"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import gVar, gData
            

class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidirectional, n_layers, noise_radius=0.2):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius=noise_radius
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        assert type(self.bidirectional)==bool
        
        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): 
            if w.dim()>1:
                weight_init.orthogonal_(w)
                
    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad
    
    def forward(self, inputs, input_lens=None, noise=False): 
        if self.embedding is not None:
            inputs=self.embedding(inputs) 
        
        batch_size, seq_len, emb_size=inputs.size()
        inputs=F.dropout(inputs, 0.5, self.training)
        
        if input_lens is not None:
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        init_hidden = gVar(torch.zeros(self.n_layers*(1+self.bidirectional), batch_size, self.hidden_size))
        hids, h_n = self.rnn(inputs, init_hidden) 
        if input_lens is not None:
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1+self.bidirectional), batch_size, self.hidden_size) 
        h_n = h_n[-1]
        enc = h_n.transpose(1,0).contiguous().view(batch_size,-1) 
        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()),std=self.noise_radius))
            enc = enc + gauss_noise
            
        return enc, hids
    
class ContextEncoder(nn.Module):
    def __init__(self, utt_encoder, input_size, hidden_size, n_layers=1, noise_radius=0.2, dia_len=10):
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.noise_radius=noise_radius
        
        self.n_layers = n_layers
        
        self.utt_encoder=utt_encoder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        

        # Added

        self.mlp_aij_mij = torch.nn.ModuleList([])

        for i in range(1, dia_len + 1):
            self.mlp_aij_mij.append(
                nn.Sequential(
                    # batch_size, max_contiext_len, max_context_len, hidden_size * 4 -> batch_size, max_context_len, max_context_len, hidden_size
                    nn.Linear((input_size - 2) * 2, hidden_size),
                    nn.BatchNorm2d(i, eps=1e-05, momentum=0.1),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm2d(i, eps=1e-05, momentum=0.1),
                    nn.ReLU(),
                    # batch_size, max_context_len, max_context_len, hidden_size -> batch_size, max_context_len, max_context_len, 1
                    nn.Linear(hidden_size, 1),
                )
            )
            self.mlp_aij_mij[i - 1].apply(self.init_weights_squential)

        # self.mlp_aij_mij = nn.Sequential(
        #     # batch_size, max_context_len, max_context_len, hidden_size * 4 -> batch_size, max_context_len, max_context_len, hidden_size
        #     nn.Linear((input_size - 2) * 2, hidden_size),
        #     nn.BatchNorm2d(hidden_size, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.BatchNorm2d(hidden_size, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     # batch_size, max_context_len, max_context_len, hidden_size -> batch_size, max_context_len, max_context_len, 1
        #     nn.Linear(hidden_size, 1),
        # )

        # self.mlp_aij_mij.apply(self.init_weights_squential)

        # Added to softplus
        self.e = 0.01

        self.mlp_sij = torch.nn.ModuleList([])

        for i in range(1, dia_len + 1):
            self.mlp_sij.append(
                nn.Sequential(
                    # batch_size, max_context_len, max_context_len, hidden_size * 4 -> batch_size, max_context_len, max_context_len, hidden_size
                    nn.Linear((input_size - 2) * 2, hidden_size),
                    nn.BatchNorm2d(i, eps=1e-05, momentum=0.1),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm2d(i, eps=1e-05, momentum=0.1),
                    nn.Tanh(),
                )
            )
            self.mlp_sij[i - 1].apply(self.init_weights_squential)

        # self.mlp_sij = nn.Sequential(
        #     # batch_size, max_context_len, max_context_len, hidden_size * 4 -> batch_size, max_context_len, max_context_len, hidden_size
        #     nn.Linear((input_size - 2) * 2, hidden_size),
        #     nn.BatchNorm2d(hidden_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.BatchNorm2d(hidden_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        # )

        self.convert_to_mu = nn.Linear(hidden_size, 1)
        self.convert_to_sigma = nn.Linear(hidden_size, 1)

        self.init_weights_squential(self.convert_to_mu)
        self.init_weights_squential(self.convert_to_sigma)

        self.update_hidden_state = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
        
        # Added
        for w in self.update_hidden_state.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
    
    def init_weights_squential(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)
    
    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, context, context_lens, utt_lens, floors, noise=False, anchor=torch.tensor([])): 
        batch_size, max_context_len, max_utt_len = context.size()
        utts=context.view(-1, max_utt_len) 
        utt_lens=utt_lens.view(-1)
        utt_encs,_ = self.utt_encoder(utts, utt_lens) 
        utt_encs = utt_encs.view(batch_size, max_context_len, -1)

        # floor_one_hot = gVar(torch.zeros(floors.numel(), 2))
        # floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
        # floor_one_hot = floor_one_hot.view(-1, max_context_len, 2)
        # utt_floor_encs = torch.cat([utt_encs, floor_one_hot], 2) 
        
        # utt_floor_encs=F.dropout(utt_floor_encs, 0.25, self.training)
        # context_lens_sorted, indices = context_lens.sort(descending=True)
        # utt_floor_encs = utt_floor_encs.index_select(0, indices)
        # utt_floor_encs = pack_padded_sequence(utt_floor_encs, context_lens_sorted.data.tolist(), batch_first=True)
        
        # init_hidden=gVar(torch.zeros(1, batch_size, self.hidden_size))
        # hids, h_n = self.rnn(utt_floor_encs, init_hidden)
        
        # _, inv_indices = indices.sort()
        # h_n = h_n.index_select(1, inv_indices)  
        
        # enc = h_n.transpose(1,0).contiguous().view(batch_size, -1)

        # Adopt message passing
        # Compute mij
        # batch_size, max_context_len, max_context_len, 2 * hidden_size
        utt_encs_i = utt_encs.unsqueeze(2).repeat(1, 1, max_context_len, 1)
        utt_encs_j = gVar(torch.zeros(batch_size, max_context_len, max_context_len, self.hidden_size * 2))
        for i in range(max_context_len):
            utt_encs_j[:, i, :, :] = utt_encs.clone()

        # batch_size, max_context_len, max_context_len, 4 * hidden_size
        utt_encs_ij = torch.cat([utt_encs_i, utt_encs_j], 3)
        aij_mij = self.mlp_aij_mij[max_context_len - 1](utt_encs_ij)

        # Compute mij(1 - mij)
        # batch_size, max_context_len, max_context_len, 1 -> batch_size, max_context_len, max_context_len
        aij_mij = aij_mij.squeeze(3)

        aij_mu = F.softplus(aij_mij) + self.e

        aij_1minusmij_positive = F.softplus((1 - aij_mu) * aij_mu) + self.e
        aij_std = torch.sqrt(aij_1minusmij_positive)

        # sample weight, aij can be negative at this step
        aij_epsilon = gVar(torch.randn([batch_size, max_context_len, max_context_len]))
        aij = F.softplus(aij_epsilon * aij_std + aij_mu) + self.e

        # Compute sij
        # batch_size, max_context_len, max_context_len, hidden_size
        sij_latent = self.mlp_sij[max_context_len - 1](utt_encs_ij)
        # batch_size, max_context_len, max_context_len, 1
        sij_mu = self.convert_to_mu(sij_latent)
        sij_mu = sij_mu.squeeze(3)
        sij_mean = aij * sij_mu

        # batch_size, max_context_len, max_context_len, 1
        sij_sigma = self.convert_to_sigma(sij_latent)
        sij_sigma = sij_sigma.squeeze(3)

        # aij is postivie, and therefore the product is positive
        sij_std = torch.sqrt(aij * sij_sigma * sij_sigma)

        sij_epsilon = gVar(torch.randn([batch_size, max_context_len, max_context_len]))
        sij = sij_epsilon * sij_std + sij_mean

        # Compute weight, sij can be negative, wij can be negative

        # print("aij size", aij.size())
        # print("sij size", sij.size())
        # print("context length", max_context_len)

        wij = aij * sij
        # Excludes later utterances and empty utterances
        # wij = wij.tril(diagonal=-1)

        enc = []


        # More neighbours
        for dialog in range(batch_size):
            # To be adjusted. Adjust floor to the previous conversation index
            # anchor = int(context_lens[dialog] - 1)
            num_neighbours = int(context_lens[dialog] - 1)

            if num_neighbours <= 0:
                enc.append(utt_encs[dialog, current_anchor, :self.hidden_size] + utt_encs[dialog, current_anchor, self.hidden_size:])
                continue

            current_anchor = int(context_lens[dialog] - 1) if anchor.size()[0] == 0 else int(anchor[dialog])
            if current_anchor < 0 or current_anchor > int(context_lens[dialog] - 1):
                current_anchor = int(context_lens[dialog] - 1)
                
            # num_neighbours
            w_anchor_removed = torch.cat([wij[dialog, current_anchor, :current_anchor], wij[dialog, current_anchor, current_anchor+1:num_neighbours+1]])
            masked_weight = F.softmax(w_anchor_removed, dim=0)
           
            # num_neighbours, hidden_size * 2
            if current_anchor == num_neighbours:
                h_anchor_removed = utt_encs[dialog, :current_anchor, :]
            elif current_anchor == 0:
                h_anchor_removed = utt_encs[dialog, current_anchor+1:num_neighbours+1, :]
            else:
                h_anchor_removed = torch.cat([utt_encs[dialog, :current_anchor, :], utt_encs[dialog, current_anchor+1:num_neighbours+1, :]], 0)

            weighted_utts = masked_weight.unsqueeze(1).expand(num_neighbours, (self.hidden_size * 2)) * h_anchor_removed
            # hidden_size * 2
            message = torch.sum(weighted_utts, dim=0)
            # 1, 1, hidden_size * 2
            message = message.unsqueeze(0).unsqueeze(0)

            initial_hidden = utt_encs[dialog, current_anchor, :self.hidden_size] + utt_encs[dialog, current_anchor, self.hidden_size:]
            # 1, 1, hidden_size 
            initial_hidden = initial_hidden.unsqueeze(0).unsqueeze(0)
            his, h_n = self.update_hidden_state(message, initial_hidden)
            enc.append(h_n.squeeze(0).squeeze(0))








        # # Original
        # for dialog in range(batch_size):
        #     # To be adjusted. Adjust floor to the previous conversation index
        #     # anchor = int(context_lens[dialog] - 1)
        #     current_anchor = int(context_lens[dialog] - 1) if anchor.size()[0] == 0 else int(anchor[dialog])
        #     if current_anchor < 0 or current_anchor > int(context_lens[dialog] - 1):
        #         current_anchor = int(context_lens[dialog] - 1)

        #     if current_anchor <= 0:
        #         # The first utterance will not be adjusted with message
        #         enc.append(utt_encs[dialog, current_anchor, :self.hidden_size] + utt_encs[dialog, current_anchor, self.hidden_size:])
        #         continue

        #             # h_z = hidden_node_states[s][:, :, z]
        #             # h_v = hidden_node_states[s]

        #             # # Sum up messages from different nosdes according to weights
        #             # m_z = softmax_pred_adj_mat[:, z, :].unsqueeze(1).expand_as(h_v) * h_v
        #             # m_z = torch.sum(m_z, dim=2)

        #             # # h_z^s = U(h_z^(s-1), m_z^s)
        #             # # Add temporal dimension
        #             # h_z = self.update_fun(h_z.unsqueeze(0).contiguous(), m_z.unsqueeze(0))
        #             # hidden_node_states[s+1][:, :, z] = h_z

        #             # if s == self.e_step - 1:
        #             #     pred_node_feats[t+1][:, :, z] = h_z.squeeze(0)
            
        #     # Excludes later utterances and empty utterances
        #     # mask = wij[dialog, anchor, :context_lens[dialog]].ne(0)

        #     # w_anchor_removed = torch.cat([wij[dialog, current_anchor, :current_anchor], wij[dialog, current_anchor, current_anchor+1:]])
        #     # masked_weight = F.softmax(w_anchor_removed, dim=0)
        #     masked_weight = F.softmax(wij[dialog, current_anchor, :current_anchor], dim=0)
        #     # anchor, hidden_size * 2
        #     weighted_utts = masked_weight.unsqueeze(1).expand(current_anchor, (self.hidden_size * 2)) * utt_encs[dialog, :current_anchor, :] 
        #     # hidden_size * 2
        #     message = torch.sum(weighted_utts, dim=0)
        #     # 1, 1, hidden_size * 2
        #     message = message.unsqueeze(0).unsqueeze(0)

        #     initial_hidden = utt_encs[dialog, current_anchor, :self.hidden_size] + utt_encs[dialog, current_anchor, self.hidden_size:]
        #     # 1, 1, hidden_size 
        #     initial_hidden = initial_hidden.unsqueeze(0).unsqueeze(0)
        #     his, h_n = self.update_hidden_state(message, initial_hidden)
        #     enc.append(h_n.squeeze(0).squeeze(0))







        # batch_size, hidden_size
        enc = torch.stack(enc, dim=0)

        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()),std=self.noise_radius))
            enc = enc + gauss_noise
        return enc
    
class Variation(nn.Module):
    def __init__(self, input_size, z_size):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size=z_size   
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu=nn.Linear(z_size, z_size) # activation???
        self.context_to_logsigma=nn.Linear(z_size, z_size) 
        
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size,_=context.size()
        context = self.fc(context)
        mu=self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context) 
        std = torch.exp(0.5 * logsigma)
        
        epsilon = gVar(torch.randn([batch_size, self.z_size]))
        z = epsilon * std + mu  
        return z, mu, logsigma 
    

class MixVariation(nn.Module):
    def __init__(self, input_size, z_size, n_components, gumbel_temp=0.1):
        super(MixVariation, self).__init__()
        self.input_size = input_size
        self.z_size=z_size  
        self.n_components = n_components
        self.gumbel_temp=0.1
        
        self.pi_net = nn.Sequential(
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, n_components),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu=nn.Linear(z_size, n_components*z_size) # activation???
        self.context_to_logsigma=nn.Linear(z_size, n_components*z_size) 
        self.pi_net.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.05, 0.05)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size,_=context.size()
        context = self.fc(context)
        
        pi=self.pi_net(context) 
        pi=F.gumbel_softmax(pi, tau=self.gumbel_temp, hard=True, eps=1e-10)
        pi=pi.unsqueeze(1) 
    
        mus=self.context_to_mu(context)
        logsigmas = self.context_to_logsigma(context) 
        stds = torch.exp(0.5 * logsigmas)
        
        epsilons = gVar(torch.randn([batch_size, self.n_components*self.z_size]))
        
        zi = (epsilons * stds + mus).view(batch_size, self.n_components, self.z_size)
        z = torch.bmm(pi, zi).squeeze(1)  # [batch_sz x z_sz]
        mu = torch.bmm(pi, mus.view(batch_size, self.n_components, self.z_size))
        logsigma = torch.bmm(pi, logsigmas.view(batch_size, self.n_components, self.z_size))
        return z, mu, logsigma
    
    
class Decoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, vocab_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size= input_size 
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size 

        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        for w in self.rnn.parameters():
            if w.dim()>1:
                weight_init.orthogonal_(w)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.fill_(0)
    
    def forward(self, init_hidden, context=None, inputs=None, lens=None):
        batch_size, maxlen = inputs.size()
        if self.embedding is not None:
            inputs = self.embedding(inputs)
        if context is not None:
            repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1)
            inputs = torch.cat([inputs, repeated_context], 2)
        inputs = F.dropout(inputs, 0.5, self.training)  
        hids, h_n = self.rnn(inputs, init_hidden.unsqueeze(0))        
        decoded = self.out(hids.contiguous().view(-1, self.hidden_size))# reshape before linear over vocab
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded
    
    def sampling(self, init_hidden, context, maxlen, SOS_tok, EOS_tok, mode='greedy'):
        batch_size=init_hidden.size(0)
        decoded_words = np.zeros((batch_size, maxlen), dtype=np.int)
        sample_lens=np.zeros(batch_size, dtype=np.int)         
     
        decoder_input = gVar(torch.LongTensor([[SOS_tok]*batch_size]).view(batch_size,1))
        decoder_input = self.embedding(decoder_input) if self.embedding is not None else decoder_input 
        decoder_input = torch.cat([decoder_input, context.unsqueeze(1)],2) if context is not None else decoder_input
        decoder_hidden = init_hidden.unsqueeze(0)        
        for di in range(maxlen):
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            decoder_output=self.out(decoder_output)
            if mode=='greedy':
                topi = decoder_output[:,-1].max(1, keepdim=True)[1] 
            elif mode=='sample':
                topi = torch.multinomial(F.softmax(decoder_output[:,-1], dim=1), 1)                    
            decoder_input = self.embedding(topi) if self.embedding is not None else topi
            decoder_input = torch.cat([decoder_input, context.unsqueeze(1)],2) if context is not None else decoder_input
            ni = topi.squeeze().data.cpu().numpy() 
            decoded_words[:,di]=ni
                      
        for i in range(batch_size):
            for word in decoded_words[i]:
                if word == EOS_tok:
                    break
                sample_lens[i]=sample_lens[i]+1
        return decoded_words, sample_lens
    

    
