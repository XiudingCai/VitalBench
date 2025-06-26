import torch
from torch import nn
from layers.Embed import PositionalEmbedding
from einops.layers.torch import Rearrange
from einops import rearrange


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.arrange_mode = configs.arrange_mode
        # padding = stride
        # padding = configs.patch_len - 1

        # patching and embedding
        # self.patch_embedding = PatchEmbedding(
        #     configs.d_model, patch_len, stride, padding, configs.dropout)
        # self.padding_patch_layer = nn.ReplicationPad1d((padding, 0))
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Sequential(
            Rearrange('b c l d -> (b c) l d'),
            nn.Linear(configs.patch_len, configs.d_model, bias=False),
            # Rearrange('(b c) l d -> b (l c) d', c=configs.enc_in),
        )
        self.position_embedding = PositionalEmbedding(configs.d_model)

        self.RevIN_mode = configs.RevIN_mode
        if self.RevIN_mode == 2:
            from models.PatchTST_OF import RevIN
            self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        print(configs.enc_in)

        from layers.mamba_ssm.mixer2_seq_simple import MixerTSModel as Mamba
        self.encoder = Mamba(
            d_model=configs.d_model,  # Model dimension d_model
            n_layer=configs.e_layers,
            n_vars=configs.enc_in,
            # dropout=0,
            dropout=configs.dropout,
            ssm_cfg={'layer': 'Mamba1'},
            arrange_mode=configs.arrange_mode,
            mamba_mode=configs.mamba_mode,
            shuffle_mode=configs.shuffle_mode,  # if self.training else 0,
            # shuffle_mode=configs.shuffle_mode if self.training else 0,
            use_casual_conv=configs.use_casual_conv,
            fused_add_norm=True,
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - configs.patch_len) / configs.stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if configs.arrange_mode == 0:
                self.head = nn.Sequential(
                    # Rearrange('b (c l) d -> (b c) (l d)', c=configs.enc_in),
                    nn.Linear(configs.d_model * configs.seq_len // configs.patch_len, configs.pred_len),
                    # Rearrange('(b c) p -> b p c', c=configs.enc_in),
                )
            else:
                self.head = nn.Sequential(
                    # Rearrange('b (l c) d -> (b c) (l d)', c=configs.enc_in),
                    nn.Linear(configs.d_model * configs.seq_len // configs.patch_len, configs.pred_len),
                    # Rearrange('(b c) p -> b p c', c=configs.enc_in),
                )

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.d_model, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            padding = self.stride - self.seq_len % self.stride if self.seq_len % self.stride != 0 else 0
            if padding != self.stride:
                self.padding_layer = nn.ReplicationPad1d((0, padding))
            else:
                self.padding_layer = None

            neck_dim = 4
            if configs.arrange_mode == 0:
                self.head = nn.Sequential(
                    Rearrange('b (c l) d -> (b c) (l d)', c=configs.enc_in),
                    nn.Linear(configs.d_model * (configs.seq_len + padding) // self.stride, neck_dim),
                    Rearrange('(b c) p -> b (p c)', c=configs.enc_in),
                    nn.Linear(configs.enc_in * neck_dim, configs.num_class)
                )
            else:
                self.head = nn.Sequential(
                    Rearrange('b (l c) d -> (b c) (l d)', c=configs.enc_in),
                    nn.Linear(configs.d_model * (configs.seq_len + padding) // self.stride, neck_dim),
                    Rearrange('(b c) p -> b (p c)', c=configs.enc_in),
                    nn.Linear(configs.enc_in * neck_dim, configs.num_class)
                )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print("input", x_enc.shape)
        b, _, n_vars = x_enc.shape

        # Norn
        if self.RevIN_mode == 1:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev
        elif self.RevIN_mode == 2:
            x_enc = self.revin_layer(x_enc, 'norm')

        # do patching and embedding
        x = x_enc.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        enc_in = self.value_embedding(x)

        # Variable Scan along Time (VST)
        if self.arrange_mode == 0:
            enc_in = rearrange(enc_in, '(b c) l d -> b (c l) d', c=n_vars)
        else:
            enc_in = rearrange(enc_in, '(b c) l d -> b (l c) d', c=n_vars)

        # Encoder
        enc_out, attns = self.encoder(enc_in)

        if self.arrange_mode == 0:
            enc_out = rearrange(enc_out, 'b (c l) d -> (b c) (l d)', c=n_vars)
        else:
            enc_out = rearrange(enc_out, 'b (l c) d -> (b c) (l d)', c=n_vars)

        # print("enc_out", enc_out.shape)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = rearrange(dec_out, '(b c) p -> b p c', c=n_vars)

        # print("dec_out", dec_out.shape)

        # De-norm
        if self.RevIN_mode == 1:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, [0], :].repeat(1, self.pred_len, 1))
        elif self.RevIN_mode == 2:
            dec_out = self.revin_layer(dec_out, 'denorm')

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

    def batch_update_state(self, cost_tensor):
        self.encoder.batch_update_state(cost_tensor)

    def set_reordering_index(self, reordering_index):
        self.encoder.set_reordering_index(reordering_index)

    def reset_ids_shuffle(self):
        self.encoder.reset_ids_shuffle()
