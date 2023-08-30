import torch.nn as nn
import torch as torch
import torch.nn.functional as F

##############################################################
#                    USED MODEL IN PREDICTIONS               #
##############################################################

class BiLSTM_W2v(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.n_layers = 4
        self.n_hidden = 100
        self.hidden_dim2 = 200
        self.hidden_dim3 = 50
        self.lstm = nn.LSTM(input_size=self.shape, hidden_size=self.n_hidden,num_layers = self.n_layers,batch_first=True, dropout=0.2, bidirectional = True)
       
        self.prediction_head = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim3, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_words):
        output, (hidden, cell) = self.lstm(input_words)
        
        # hidden[-2,:,:] este starea finala la parcurgerea inainte
        # hidden[-1,:,:] este starea finala la parcurgerea inapoi

        prediction_out = self.prediction_head(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return prediction_out


##############################################################
#                           EXPERIMENTS                      #
##############################################################


class LSTM_1L(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.n_layers = 3   # number of LSTM layers 
        self.n_hidden = 100   # number of hidden nodes in LSTM
        self.hidden_dim2 = 200
        self.hidden_dim3 = 50

        # self.embedding = nn.Embedding.from_pretrained(vocab_vectors)
        # self.embedding.weight.requires_grad = False

        # self.lstm = nn.LSTM(input_size=self.shape, hidden_size=self.n_hidden,num_layers = self.n_layers,batch_first=True, dropout=0.5)
        # self.embedding = nn.Embedding(shape+1, 100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=self.n_hidden,num_layers = self.n_layers,batch_first=True, dropout=0.2)
        self.prediction_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),  # 160 /16 = 9 * 64 = 576
            nn.BatchNorm1d(self.hidden_dim3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim3, 2),  # 160 /16 = 9 * 64 = 576
            nn.Softmax(dim=1)
        )

    def forward(self, input_words):
        # self.init_hidden(16)
        # embedded_words = self.embedding(input_words)
        _ , lstm_out  = self.lstm(input_words)         # (batch_size, seq_length, n_hidden)
        h_n = lstm_out[0]
        c_n = lstm_out[1]
        # lstm_out = torch.permute(c_n, (1, 0, 2))
        # lstm_out = h_n[-1].reshape((h_n[-1].shape[0], -1))
        lstm_out = torch.cat((
            c_n[-1].reshape((c_n[-1].shape[0], -1)),
            h_n[-1].reshape((h_n[-1].shape[0], -1))
            ), dim=1)

        # lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)
        sigmoid_out = self.prediction_head(lstm_out)
        # sigmoid_out = sigmoid_out.view(16, -1)
        # extract the output of ONLY the LAST output of the LAST element of the sequence
        # sigmoid_last = sigmoid_out[:, -1]
        return sigmoid_out

    def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h

class LSTM_2L(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.n_embed = 100
        self.n_layers = 2
        self.n_hidden = 64     # nobody cares
        self.hidden_dim2 = 128  # 2 * n_hidden
        self.hidden_dim3 = 32

        # self.lstm = nn.LSTM(input_size=self.shape, hidden_size=self.n_hidden,num_layers = self.n_layers,batch_first=True, dropout=0.5)
        self.embedding = nn.Embedding(shape, self.n_embed)
        self.lstm1 = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_embed, num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_hidden, num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.prediction_head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),  # 160 /16 = 9 * 64 = 576
            nn.BatchNorm1d(self.hidden_dim3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim3, 2),  # 160 /16 = 9 * 64 = 576
            nn.Softmax(dim=1)
        )

    def forward(self, input_words):
        embedded_words = self.embedding(input_words)
        lstm1_out ,_  = self.lstm1(embedded_words)         # (batch_size, seq_length, n_hidden)
        lstm2_in = lstm1_out + embedded_words
        _, lstm_out  = self.lstm2(lstm2_in)
        h_n = lstm_out[0]
        c_n = lstm_out[1]
        # lstm_out = torch.permute(c_n, (1, 0, 2))
        #TODO concat cu mean de media de output(_)
        lstm_out = torch.cat((
            c_n[-1].reshape((c_n[-1].shape[0], -1)),
            h_n[-1].reshape((h_n[-1].shape[0], -1))
            ), dim=1)
        sigmoid_out = self.prediction_head(lstm_out)
        return sigmoid_out

    # def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     weights = next(self.parameters()).data
    #     h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
    #          weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
    #     return h

class LSTM_5ls(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.n_embed = 100 #256
        self.n_layers = 4   # number of LSTM layers 
        self.shape = shape
        self.n_hidden = 50#512   # number of hidden nodes in LSTM
        self.hidden_dim2 = 100#1024 # 2 * n_hidden
        self.hidden_dim3 = 50#256

        self.embedding = nn.Embedding(shape+1, self.n_embed)
        self.lstm1 = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_embed, num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_embed, num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_embed, num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.lstm4 = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_embed, num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.lstm5 = nn.LSTM(input_size=self.n_embed, hidden_size=self.n_hidden, num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.prediction_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),  # 160 /16 = 9 * 64 = 576
            # nn.BatchNorm1d(self.hidden_dim3),
            nn.LeakyReLU(0.1),
            # nn.Dropout(0.3),
            nn.Linear(self.hidden_dim3, 2),  # 160 /16 = 9 * 64 = 576
            nn.Softmax(dim=1)
        )


    def forward(self, input_words):
        embedded_words = self.embedding(input_words)
        lstm1_out ,_  = self.lstm1(embedded_words)         # (batch_size, seq_length, n_hidden)
        lstm2_in = lstm1_out + embedded_words

        lstm2_out ,_  = self.lstm2(lstm2_in)         # (batch_size, seq_length, n_hidden)
        lstm3_in = lstm2_out + lstm2_in

        lstm3_out ,_  = self.lstm3(lstm3_in)         # (batch_size, seq_length, n_hidden)
        lstm4_in = lstm3_out + lstm3_in

        lstm4_out ,_  = self.lstm4(lstm4_in)         # (batch_size, seq_length, n_hidden)
        lstm5_in = lstm4_out + lstm4_in
        
        _, lstm_out  = self.lstm5(lstm5_in)
        h_n = lstm_out[0]
        c_n = lstm_out[1]
        # lstm_out = torch.permute(c_n, (1, 0, 2))
        lstm_out = torch.cat((
            c_n[-1].reshape((c_n[-1].shape[0], -1)),
            h_n[-1].reshape((h_n[-1].shape[0], -1))
            ), dim=1)
        sigmoid_out = self.prediction_head(lstm_out)
        return sigmoid_out


class LSTM_W2v(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.n_layers = 4   # number of LSTM layers 
        self.n_hidden = 100   # number of hidden nodes in LSTM
        self.hidden_dim2 = 200
        self.hidden_dim3 = 50

        # self.embedding = nn.Embedding.from_pretrained(vocab_vectors)
        # self.embedding.weight.requires_grad = False

        # self.lstm = nn.LSTM(input_size=self.shape, hidden_size=self.n_hidden,num_layers = self.n_layers,batch_first=True, dropout=0.5)
        self.lstm1 = nn.LSTM(input_size=self.shape, hidden_size=self.shape,     num_layers = self.n_layers, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=self.shape, hidden_size=self.shape,     num_layers = self.n_layers, batch_first=True, dropout=0.1)
        self.lstm3 = nn.LSTM(input_size=self.shape, hidden_size=self.shape,     num_layers = self.n_layers, batch_first=True, dropout=0.1)
        self.lstm4 = nn.LSTM(input_size=self.shape, hidden_size=self.shape,     num_layers = self.n_layers, batch_first=True, dropout=0.1)
        self.lstm5 = nn.LSTM(input_size=self.shape, hidden_size=self.n_hidden,  num_layers = self.n_layers, batch_first=True, dropout=0.1)
        self.prediction_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),  # 160 /16 = 9 * 64 = 576
            nn.BatchNorm1d(self.hidden_dim3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim3, 2),  # 160 /16 = 9 * 64 = 576
            nn.Softmax(dim=1)
        )

    def forward(self, input_words):
        # self.init_hidden(16)

        lstm1_out ,_  = self.lstm1(input_words)         # (batch_size, seq_length, n_hidden)
        # lstm1_out ,_  = self.lstm1(input_words.float())         # (batch_size, seq_length, n_hidden)
        lstm2_in = lstm1_out + input_words

        lstm2_out ,_  = self.lstm2(lstm2_in)         # (batch_size, seq_length, n_hidden)
        # lstm2_out ,_  = self.lstm2(lstm2_in.float())         # (batch_size, seq_length, n_hidden)
        lstm3_in = lstm2_out + lstm2_in

        lstm3_out ,_  = self.lstm3(lstm3_in)         # (batch_size, seq_length, n_hidden)
        # lstm3_out ,_  = self.lstm3(lstm3_in.float())         # (batch_size, seq_length, n_hidden)
        lstm4_in = lstm3_out + lstm3_in

        lstm4_out ,_  = self.lstm4(lstm4_in)         # (batch_size, seq_length, n_hidden)
        # lstm4_out ,_  = self.lstm4(lstm4_in.float())         # (batch_size, seq_length, n_hidden)
        lstm5_in = lstm4_out + lstm4_in
        
        _, lstm_out  = self.lstm5(lstm5_in)
        # _,lstm_out   = self.lstm5(lstm5_in.float())
        h_n = lstm_out[0]
        c_n = lstm_out[1]
        # lstm_out = torch.permute(c_n, (1, 0, 2))
        #TODO concat cu mean de media de output(_)
        lstm_out = torch.cat((
            c_n[-1].reshape((c_n[-1].shape[0], -1)),
            h_n[-1].reshape((h_n[-1].shape[0], -1))
            ), dim=1)
        sigmoid_out = self.prediction_head(lstm_out)
        # sigmoid_out = self.prediction_head(lstm_out.float())
        # sigmoid_out = self.prediction_head(lstm_out[:, -1, :].float())
        return sigmoid_out
    
    def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h


class LSTM_W2v_2L(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.n_layers = 4   # number of LSTM layers 
        self.n_hidden = 200   # number of hidden nodes in LSTM
        self.hidden_dim2 = 400
        self.hidden_dim3 = 100

        # self.embedding = nn.Embedding.from_pretrained(vocab_vectors)
        # self.embedding.weight.requires_grad = False

        # self.lstm = nn.LSTM(input_size=self.shape, hidden_size=self.n_hidden,num_layers = self.n_layers,batch_first=True, dropout=0.5)
        self.lstm1 = nn.LSTM(input_size=self.shape, hidden_size=self.shape,     num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=self.shape, hidden_size=self.n_hidden,     num_layers = self.n_layers, batch_first=True, dropout=0.2)
        self.prediction_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),  # 160 /16 = 9 * 64 = 576
            nn.BatchNorm1d(self.hidden_dim3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim3, 2),  # 160 /16 = 9 * 64 = 576
            nn.Softmax(dim=1)
        )

    def forward(self, input_words):
        # self.init_hidden(16)

        lstm1_out ,_  = self.lstm1(input_words)         # (batch_size, seq_length, n_hidden)
        # lstm1_out ,_  = self.lstm1(input_words.float())         # (batch_size, seq_length, n_hidden)
        lstm2_in = lstm1_out + input_words
        
        _, lstm_out  = self.lstm2(lstm2_in)
        # _,lstm_out   = self.lstm5(lstm5_in.float())
        h_n = lstm_out[0]
        c_n = lstm_out[1]
        # lstm_out = torch.permute(c_n, (1, 0, 2))
        #TODO concat cu mean de media de output(_)
        lstm_out = torch.cat((
            c_n[-1].reshape((c_n[-1].shape[0], -1)),
            h_n[-1].reshape((h_n[-1].shape[0], -1))
            ), dim=1)
        sigmoid_out = self.prediction_head(lstm_out)
        # sigmoid_out = self.prediction_head(lstm_out.float())
        # sigmoid_out = self.prediction_head(lstm_out[:, -1, :].float())
        return sigmoid_out
    
    def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h
    

class CNN(nn.Module):
    def __init__(self, shape, num_filters, filter_sizes, hidden_dim, output_dim, dropout):
        super().__init__()

        # self.embedding = nn.Embedding(shape+1, 100)
        # self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, 100)) for fs in filter_sizes])
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, shape)) for fs in filter_sizes])
        self.batch = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.soft = nn.Softmax(dim=1)

    def forward(self, embedded):
        # embedded = self.embedding(text) # embedded shape: [batch_size, max_len, embedding_dim]
        text = embedded.unsqueeze(1)  # add channel dimension
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.conv_layers]
        pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        fc1_out = F.relu(self.dropout(self.fc1(cat)))
        fc2_out = self.fc2(self.batch(fc1_out))

        return self.soft(fc2_out)
    


class CNN_LSTM(nn.Module):
    def __init__(self, n_filters, filter_sizes, hidden_dim, output_dim, dropout):
        super().__init__()
        self.n_layers = 4
        self.n_hidden = 100
        self.hidden_dim2 = 200
        self.hidden_dim3 = 50
       
        self.convs = nn.ModuleList([nn.Conv2d(1, n_filters, (fs, 100)) for fs in filter_sizes])
        # self.lstm = nn.LSTM(n_filters * len(filter_sizes), hidden_dim, bidirectional=True, num_layers=4, dropout=dropout)
        self.lstm = nn.LSTM(input_size=n_filters * len(filter_sizes), hidden_size=self.n_hidden,num_layers = self.n_layers,batch_first=True, dropout=0.2, bidirectional = True)
        self.batch = nn.BatchNorm1d(2 * hidden_dim)

        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.soft = nn.Softmax(dim=1)
        self.prediction_head = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim2),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim3, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, text):
        text = text.unsqueeze(1)  
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]  
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  
        cat = self.dropout(torch.cat(pooled, dim=1))
        cat = cat.view(len(cat), 1, -1)
        output, (hidden, cell) = self.lstm(cat)
        # lstm_out = lstm_out.view(len(lstm_out), -1)
        # out = self.prediction_head(lstm_out)
        prediction_out = self.prediction_head(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        # out = self.fc(self.batch(lstm_out))
        return prediction_out