from modules.Layer import *


class LSTMscorer(nn.Module):
    def __init__(self, vocab, config):
        super(LSTMscorer, self).__init__()
        self.config = config

        self.lstm = MyLSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.score = nn.Linear(in_features=config.lstm_hiddens * 2,
                               out_features=vocab.label_size,
                               bias=False)

        torch.nn.init.kaiming_uniform_(self.score.weight)

    def forward(self, char_represents, char_mask):
        lstm_hidden, _ = self.lstm(char_represents, char_mask, None)
        lstm_hidden= lstm_hidden.transpose(1, 0)
        score = self.score(lstm_hidden)
        return score



