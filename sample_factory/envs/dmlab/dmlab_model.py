import torch

from torch import nn

from sample_factory.algorithms.appo.model_utils import create_standard_encoder, EncoderBase, register_custom_encoder
from sample_factory.envs.dmlab.dmlab30 import DMLAB_VOCABULARY_SIZE, DMLAB_INSTRUCTIONS
from sample_factory.utils.utils import log
from sample_factory.algorithms.appo.modules.meta_modules import DmlabEncoder as DmlabEncoderMeta


class DmlabEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)

        self.basic_encoder = create_standard_encoder(cfg, obs_space, timing, action_sizes)
        self.encoder_out_size = self.basic_encoder.encoder_out_size

        # same as IMPALA paper
        self.embedding_size = 20
        self.instructions_lstm_units = 64
        self.instructions_lstm_layers = 1

        padding_idx = 0
        self.word_embedding = nn.Embedding(
            num_embeddings=DMLAB_VOCABULARY_SIZE,
            embedding_dim=self.embedding_size,
            padding_idx=padding_idx
        )

        self.instructions_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.instructions_lstm_units,
            num_layers=self.instructions_lstm_layers,
            batch_first=True,
        )

        # learnable initial state?
        # initial_hidden_values = torch.normal(0, 1, size=(self.instructions_lstm_units, ))
        # self.lstm_h0 = nn.Parameter(initial_hidden_values, requires_grad=True)
        # self.lstm_c0 = nn.Parameter(initial_hidden_values, requires_grad=True)

        self.encoder_out_size += self.instructions_lstm_units
        log.debug('Policy head output size: %r', self.encoder_out_size)

        if cfg.dmlab_instr_encoder_device == 'cpu':
            self.instr_device = torch.device('cpu')
        else:
            assert torch.cuda.device_count() == 1
            self.instr_device = torch.device('cuda', index=0)

    def model_to_device(self, device):
        self.to(device)
        self.instr_device = device
        self.word_embedding.to(self.instr_device)
        self.instructions_lstm.to(self.instr_device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return self.instr_device, torch.int64
        else:
            return self.model_device(), torch.float32

    def forward_instr(self, obs_dict):
        with self.timing.timeit('dmlab_prepare_instr'):
            with torch.no_grad():
                instr = obs_dict[DMLAB_INSTRUCTIONS]
                instr_lengths = (instr != 0).sum(axis=1)
                instr_lengths = torch.clamp(instr_lengths, min=1)
                max_instr_len = torch.max(instr_lengths).item()
                instr = instr[:, :max_instr_len]
                instr_lengths_cpu = instr_lengths.to('cpu')

        with self.timing.timeit('dmlab_encode_instr'):
            instr_embed = self.word_embedding(instr)
            instr_packed = torch.nn.utils.rnn.pack_padded_sequence(
                instr_embed, instr_lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            rnn_output, _ = self.instructions_lstm(instr_packed)
            rnn_outputs, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

            first_dim_idx = torch.arange(rnn_outputs.shape[0])
            last_output_idx = sequence_lengths - 1
            last_outputs = rnn_outputs[first_dim_idx, last_output_idx]

        return last_outputs

    def forward(self, obs_dict, **kwargs):
        with self.timing.timeit('dmlab_basic_enc'):
            x = self.basic_encoder(obs_dict)

        if kwargs.get('instr', None) is None:
            ext_data = self.forward_instr(obs_dict)
        else:
            ext_data = kwargs.get('instr')

        if self.instr_device.type == 'cpu':
            with self.timing.timeit('dmlab_last_output_device'):
                ext_data = ext_data.to(x.device)  # for some reason this is very slow

        x = torch.cat((x, ext_data), dim=1)
        return x

def dmlab_register_models():
    register_custom_encoder('dmlab_instructions', DmlabEncoder)
    register_custom_encoder('dmlab_instructions_meta', DmlabEncoderMeta)
