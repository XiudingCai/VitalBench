import torch
import torch.nn as nn

from layers.Embed import PatchEmbedding
from models.ssm.s4 import S4Block as S4
from models.PatchTST import FlattenHead


class S4Net(nn.Module):
    def __init__(
            self,
            model_dim,
            dropout,
            nlayers,
            prenorm,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(nlayers):
            self.s4_layers.append(S4(model_dim, dropout=dropout, transposed=False))
            self.norms.append(nn.LayerNorm(model_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x, lengths=None):
        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        return x


class S4Model(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            model_dim,
            nlayers=4,
            dropout=0.2,
            prenorm=False,
            *args,
            **kwargs
    ):
        super().__init__()
        self.model_type = "s4_model"
        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.embed = nn.Linear(input_dim, model_dim)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(nlayers):
            self.s4_layers.append(S4(model_dim, dropout=dropout, transposed=False))
            self.norms.append(nn.LayerNorm(model_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, x, lengths=None):
        """
        Input x is shape (B, L, input_dim)
        """
        x = self.embed(x)  # (B, L, input_dim) -> (B, L, hidden_dim)

        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, hidden_dim) -> (B, output_dim)

        return x


class Model_bk(nn.Module):
    def __init__(
            self,
            configs,
            prenorm=False,
            *args,
            **kwargs
    ):
        super().__init__()
        self.model_type = "s4_model"
        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.embed = nn.Linear(configs.seq_len, configs.d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(configs.e_layers):
            self.s4_layers.append(S4(configs.d_model, dropout=configs.dropout, transposed=False))
            self.norms.append(nn.LayerNorm(configs.d_model))
            self.dropouts.append(nn.Dropout(configs.dropout))

        self.decoder = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x, lengths=None):
        """
        Input x is shape (B, L, input_dim)
        """
        x = self.embed(x)  # (B, L, input_dim) -> (B, L, hidden_dim)

        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, hidden_dim) -> (B, output_dim)

        return x


class Model(nn.Module):
    def __init__(
            self,
            configs,
            prenorm=True,
            *args,
            **kwargs
    ):
        super().__init__()
        self.model_type = "s4_model"
        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        # self.embed = nn.Linear(configs.seq_len, configs.d_model)

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = configs.stride
        self.RevIN_mode = configs.RevIN_mode

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(configs.e_layers):
            self.s4_layers.append(S4(configs.d_model, dropout=configs.dropout, transposed=False))
            self.norms.append(nn.LayerNorm(configs.d_model))
            self.dropouts.append(nn.Dropout(configs.dropout))

            # Prediction Head
            self.head_nf = configs.d_model * \
                           int((configs.seq_len - configs.patch_len) / configs.stride + 2)
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                        head_dropout=configs.dropout)
            # elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            #     self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
            #                             head_dropout=configs.dropout)
            # elif self.task_name == 'classification':
            #     self.flatten = nn.Flatten(start_dim=-2)
            #     self.dropout = nn.Dropout(configs.dropout)
            #     self.projection = nn.Linear(
            #         self.head_nf * configs.enc_in, configs.num_class)

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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Input x is shape (B, L, input_dim)
        """
        if self.RevIN_mode == 1:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        x, n_vars = self.patch_embedding(x_enc)

        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=None)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        enc_out = x.transpose(-1, -2)

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.RevIN_mode == 1:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


if __name__ == '__main__':
    # x = torch.randn(2, 96, 21).cuda()
    # model = S4Model(input_dim=21, output_dim=21, model_dim=512).cuda()
    # y = model(x)
    # print(y.shape)

    x = torch.randn(2, 96, 21).cuda()
    model = S4Model(input_dim=21, output_dim=21, model_dim=512).cuda()
    y = model(x)
    print(y.shape)
