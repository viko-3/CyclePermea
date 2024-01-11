import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import RobertaTokenizerFast, RobertaModel, AutoModelForMaskedLM, RobertaConfig
import math
import timm
from contextlib import contextmanager

from utils import extent_mask, LitEma


class BiRNNRegression(nn.Module):
    def __init__(self, vocab, model_config, data_config):
        super(BiRNNRegression, self).__init__()
        self.vocab_pad = vocab.pad
        self.vocab_size = len(vocab)
        self.embedding_dim = model_config['emb_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.output_dim = model_config['output_dim']
        self.bidirectional = model_config['bidirectional']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.vocab_pad)

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True,
                           bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # 乘以2是因为双向
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, length):
        # RNN层
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).cuda()
        x = pack_padded_sequence(x, lengths=length, batch_first=True)

        out, (h0, c0) = self.rnn(x, (h0, c0))

        out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.fc1(out[:, -1, :])
        # out = self.fc2(out)
        return out


class TextCNN(nn.Module):
    """CNN model."""

    def __init__(self, vocab, model_config, data_config):
        """Initialization of CNN model.

        :param vocab_size: vocabulary.
        :param embedding_dim: int, dimension of word e
        :param n_filters: int, count of filters
        :param filter_sizes: list, dimensions of each filter
        :param output_dim: int, output size
        :param dropout: float, rate of drop out
        :param pad_idx: int, padding index
        """
        super(TextCNN, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.emb_dim = model_config['emb_dim']
        self.n_filters = model_config['n_filters']
        self.filter_sizes = model_config['filter_sizes']
        self.output_dim = model_config['output_dim']
        self.dropout = model_config['drop_out']
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.vocab.pad)
        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=self.n_filters,
                                kernel_size=(self.filter_sizes[0], self.emb_dim))
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=self.n_filters,
                                kernel_size=(self.filter_sizes[1], self.emb_dim))
        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=self.n_filters,
                                kernel_size=(self.filter_sizes[2], self.emb_dim))
        # Full connected layer.
        self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters, self.output_dim)
        # Drop out to reduce over-fitting.
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, text, length):
        """Defines the forward propagation process.

        :param text: input data
        :return: output of model,the predictions
        """

        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)


class Transformer(nn.Module):

    def __init__(self, vocab, model_config, data_config, multi_model=False):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.max_len = data_config['max_len']
        self.pad_mask = None
        self.emb_dim = model_config['emb_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.output_dim = model_config['output_dim']
        self.nhead = model_config['num_head']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['drop_out']
        self.vocab_size = len(vocab)
        self.encoder = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=vocab.pad)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.emb_dim))
        self.drop = nn.Dropout(self.dropout)
        # transformer
        self.encoder_blocks = nn.Sequential(
            *[Block(self.emb_dim, self.hidden_dim, self.nhead, self.max_len, dropout=self.dropout,
                    multi_mode=multi_model) for _ in range(self.num_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.hidden_dim)
        # self.initialize_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_parameters(self):
        # 遍历模型的所有参数并初始化为均匀分布
        for param in self.parameters():
            if param.requires_grad:
                nn.init.uniform_(param)  # 这里的a和b是均匀分布的上下界

    def forward(self, x, lengths, image_feature=None):
        b, t = x.size()
        self.pad_mask = (x != self.vocab.pad).int()

        x = self.encoder(x)  # 字符编码+位置编码
        x = self.drop(x + self.pos_emb[:, :t, :])

        for layer in self.encoder_blocks:
            x = layer(x, self.pad_mask, image_feature)
        x = self.ln_f(x)
        return x  # batch,len,dim


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, emb_dim, hidden_dim, head, max_len, multi_mode=False, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.multi_mode = multi_mode
        self.self_attn = SelfAttention(emb_dim, hidden_dim, head, max_len,
                                       is_cross_attention=False,
                                       dropout=dropout)
        self.cross_attn = SelfAttention(emb_dim, hidden_dim, head, max_len,
                                        is_cross_attention=True,
                                        dropout=dropout) if self.multi_mode else None
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
        )

    def forward(self, x, pad_mask, image_feature=None):
        x = self.ln1(x)
        self_attention_output = self.self_attn(x, pad_mask, image_feature=None)
        x = x + self_attention_output
        if self.multi_mode and image_feature is not None:
            cross_attention_output = self.cross_attn(x, pad_mask=None, image_feature=image_feature)
            x = x + cross_attention_output
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, emb_dim, hidden_dim, head, max_len, is_cross_attention=False, dropout=0.3):
        super().__init__()
        assert hidden_dim % head == 0
        self.max_len = max_len
        # key, query, value projections for all heads
        self.key = nn.Linear(emb_dim, hidden_dim)
        self.query = nn.Linear(emb_dim, hidden_dim)
        self.value = nn.Linear(emb_dim, hidden_dim)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.n_head = head

        self.is_cross_attention = is_cross_attention

    def extend_mask(self, attention_mask):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, x, pad_mask, image_feature=None):
        B, T, C = x.size()

        if not self.is_cross_attention:
            # self-attention
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        else:
            # cross-attention
            _, _T, _C = image_feature.size()
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            k = self.key(image_feature).view(B, _T, self.n_head, _C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.value(image_feature).view(B, _T, self.n_head, _C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # batch,head,text_len,image_len

        if pad_mask is not None:
            mask = self.extend_mask(pad_mask)
            att = att + mask

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y  # , attn_save


class ChemBERTa(nn.Module):
    def __init__(self, vocab):
        super(ChemBERTa, self).__init__()
        self.vocab = vocab
        model_name = "DeepChem/ChemBERTa-77M-MLM"
        #config = RobertaConfig.from_pretrained("DeepChem/ChemBERTa-77M-MLM", output_hidden_states=True)
        self.model =   RobertaModel.from_pretrained(model_name, output_hidden_states=True,) # RobertaModel(config)

    def forward(self, x, length, image_feature=None):
        self.pad_mask = (x != self.vocab.pad_token_id).int()
        out = self.model(x)
        states = out.hidden_states[-1].squeeze()
        states = states.unsqueeze(0) if states.dim() == 2 else states  # only one data  （len，dim）-》（1，len,dim）
        return states


class PubChemLM(nn.Module):
    def __init__(self, vocab):
        super(PubChemLM, self).__init__()
        self.vocab = vocab
        model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)

    def forward(self, x, length=0, image_feature=None):
        self.pad_mask = (x != self.vocab.pad_token_id).int()
        out = self.model(x, output_attentions=False)
        # torch.save(out.attentions,'attn.pt')
        states = out.hidden_states[-1].squeeze()
        states = states.unsqueeze(0) if states.dim() == 2 else states  # only one data  （len，dim）-》（1，len,dim）
        return states


class Gate_Layer(nn.Module):
    def __init__(self, latent_feature_dim, fingerprint_dim):
        super(Gate_Layer, self).__init__()
        self.fingerprint_dim = fingerprint_dim
        self.latent_feature_fim = latent_feature_dim
        self.alignment_layer = nn.Linear(self.fingerprint_dim, self.latent_feature_fim)
        self.gate_layer = nn.Linear(self.latent_feature_fim * 2, 1)

    def forward(self, latent_feature, fingerprint):
        fingerprint = self.alignment_layer(fingerprint.float())
        tensor_for_gate = torch.cat((latent_feature, fingerprint), dim=1)
        importance_scale = F.sigmoid(self.gate_layer(tensor_for_gate))
        tensor_for_next_layer = importance_scale * latent_feature + (1 - importance_scale) * fingerprint
        return tensor_for_next_layer


class FeatureFusionModel(nn.Module):
    def __init__(self, input_dim_model=784, input_dim_fingerprint=1024):
        super(FeatureFusionModel, self).__init__()
        # Alignment Layer
        self.alignment_layer = nn.Sequential(
            nn.Linear(input_dim_fingerprint, 1000),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout
            nn.Linear(1000, input_dim_model),
            nn.BatchNorm1d(input_dim_model),
            nn.ReLU()
        )
        # Gate Layer
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim_model * 2, 500),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        # Attention Layer (Optional)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim_model, num_heads=4)

    def forward(self, tensor_from_model, tensor_from_fingerprint):
        tensor_from_fingerprint = tensor_from_fingerprint.float()

        # Feature alignment
        aligned_tensor = self.alignment_layer(tensor_from_fingerprint)

        # Attention (Optional)
        # tensor_from_model, _ = self.attention(aligned_tensor.unsqueeze(1), tensor_from_model.unsqueeze(1),
        #                                        tensor_from_model.unsqueeze(1))
        tensor_from_model = tensor_from_model.squeeze(1)

        # Concatenate feature
        tensor_for_gate = torch.cat((tensor_from_model, aligned_tensor), dim=1)

        # Gate combination
        importance_scale = self.gate_layer(tensor_for_gate)

        # Weighted sum
        tensor_for_next_layer = importance_scale * tensor_from_model + (1 - importance_scale) * aligned_tensor
        return tensor_for_next_layer


class Prediction_Head(nn.Module):
    def __init__(self, latent_feature_dim):
        super(Prediction_Head, self).__init__()
        self.latent_feature_dim = latent_feature_dim
        self.SMILES_Restruction_Block = None
        self.HELM_dim = 4
        self.fingerprint_dim = 1024
        self.HELM_Prediction_Block = nn.Sequential \
            (nn.Linear(self.latent_feature_dim, self.latent_feature_dim // 2),
             nn.GELU(),
             nn.Linear(self.latent_feature_dim // 2, self.HELM_dim * 17)  # 4个独立的分类器
             )

        # self.regresion_Prediction_Block = nn.Sequential(
        #     nn.Linear((self.latent_feature_dim + 1024), (self.latent_feature_dim + 1024) // 2),
        #     nn.GELU(),
        #     nn.Linear((self.latent_feature_dim + 1024) // 2, 1))
        self.gate_layer = FeatureFusionModel(self.latent_feature_dim, self.fingerprint_dim)
        self.regresion_Prediction_Block = nn.Sequential(
            nn.Linear(self.latent_feature_dim, self.latent_feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.latent_feature_dim // 2, 1)
        )

    def forward(self, latent_feature, fingerprint):
        # 回归任务
        logits = latent_feature[:, 0]
        # HELM预测任务
        HELM = self.HELM_Prediction_Block(logits)
        HELM = HELM.view(-1, self.HELM_dim, 17)
        if fingerprint is not None:
            # logits = torch.cat((logits, fingerprint), dim=1)
            logits = self.gate_layer(logits, fingerprint)
        logits = self.regresion_Prediction_Block(logits)

        return logits, HELM


class Peptide_Regression(nn.Module):
    def __init__(self, all_model_config, text_vocab, text_model_config, text_data_config, use_ema=True):
        super().__init__()
        self.feature_cmb_type = all_model_config['feature_cmb_type']

        self.use_text_info = all_model_config['use_text_info']
        self.use_fg_info = all_model_config['use_fingerprint_info']
        self.text_vocab = text_vocab
        self.text_model_config = text_model_config
        self.text_data_config = text_data_config
        self.use_image_info = False

        self.built_model(self.use_text_info, self.use_image_info, self.text_vocab, self.text_model_config,
                         self.text_data_config)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @property
    def device(self):
        return next(self.parameters()).device

    def built_model(self, use_text_info, use_image_info, text_vocab, text_model_config, text_data_config):

        self.predict_head = Prediction_Head(text_model_config['hidden_dim'])

        if text_model_config['model_type'] == 'Transformer':
            self.text_model = Transformer(text_vocab, text_model_config, text_data_config,
                                          multi_model=True if use_text_info and use_image_info else False)
        elif text_model_config['model_type'] == 'CNN':
            self.text_model = TextCNN(text_vocab, text_model_config, text_data_config)
            self.text_mlp = None
        elif text_model_config['model_type'] == 'LSTM':
            self.text_model = BiRNNRegression(text_vocab, text_model_config, text_data_config)
            self.text_mlp = None
        elif text_model_config['model_type'] == 'ChemBERTa':
            self.text_model = ChemBERTa(vocab=self.text_vocab)
        elif text_model_config['model_type'] == 'PubChemLM':
            self.text_model = PubChemLM(vocab=self.text_vocab)
        else:
            self.text_model = None
            print('Need True text model')

    def forward(self, text, fingerprint=None, length=0):
        if not self.use_fg_info:
            fingerprint = None
        latent_feature = self.text_model(text, length)

        logits, HELM = self.predict_head(latent_feature, fingerprint)

        return latent_feature[:, 0], logits, HELM


class SqueezeModule(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)
