import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMCells(nn.Module):
    def __init__(self, input_fea, hidden_unit, return_sequence = False, 
                                                recurrent_dropout = None):
        super(LSTMCells, self).__init__()
        self.return_sequence = return_sequence
        self.hidden_dim = hidden_unit
        self.input_fea  = input_fea
        
        self.Linear1 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear2 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear3 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear4 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        
        self.Linear5 = nn.Linear(hidden_unit, hidden_unit, bias=True)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()
                
    def forward(self, x): # [batch, width, height] --> # [batch, 1, height]
        Batch = []
        Batch_seq = []
        Seq = []
        
        # Initialize first cell state/hidden state
        c0 = torch.zeros(x.shape[0], self.hidden_dim).to(device) #.detach().to(device)
        h0 = torch.zeros(x.shape[0], self.hidden_dim).to(device) #.detach().to(device)
        
        # Here is the number of LSTM cells x.shape[1]
        for time_step in range(x.shape[1]): # self.input_fea number of timesteps
            # print('shape x: ', x.shape)
            seq = x[:, time_step, :]
            
            Input1 = torch.cat((h0, seq), -1).float().to(device)
            # print('shape input: ', Input1.shape)
            # print('shape input: ', c0.shape)
            # print('shape input: ', h0.shape)
            
            sigma1 = self.sigmoid(self.Linear1(Input1))
            sigma2 = self.sigmoid(self.Linear2(Input1))
            sigma3 = self.sigmoid(self.Linear3(Input1))
            
            mul1 = sigma1 * c0
            mul2 = sigma2 * self.tanh(self.Linear4(Input1))

            c1 = mul1 + mul2
            
            h1 = self.tanh(self.Linear5(c1)) * sigma3
            # h1 = self.tanh(c1) * sigma3
            
            c0, h0 = c1, h1

            if time_step == 0:
                Seq = h1.reshape(x.shape[0], 1, self.hidden_dim)
            if time_step > 0:
                Seq = torch.cat((Seq, h1.reshape(x.shape[0], 1, self.hidden_dim)), 1)

        if self.return_sequence == False : return h1
        if self.return_sequence == True  : return Seq