import chess
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import random

current_directory =  os.path.join(os.path.dirname(__file__), 'data')
file_chess_move = [f for f in os.listdir(current_directory) 
                   if f.endswith(f".csv")]
path = os.path.join(current_directory, file_chess_move[1])
df = pd.read_csv(path, usecols=[
    'move_no', 
    'move_no_pair', 
    'player', 
    'move', 
    'color', 
    'fen'])	
player_name = file_chess_move[1].split('-')[0]

letter_2_num = {'a':0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_2_letter = {0: 'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device configuration

def board_2_rep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layers(board, piece))
    board_rep = np.stack(layers)
    return board_rep

def create_rep_layers(board, type):
    s = str(board)
    s = re.sub(f'[^{type}{type.upper()} \n]', '.', s)
    s = re.sub(f'{type}', '-1', s)
    s = re.sub(f'{type.upper()}', '1', s)
    s = re.sub(f'\\.', '0', s)
    
    board_mat = []
    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        board_mat.append(row)
    return np.array(board_mat)

def move_2_rep(move, board):
    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8,8))
    from_row = 8 - int(move[1])
    from_column = letter_2_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8,8))
    to_row = 8 - int(move[3])
    tow_column = letter_2_num[move[2]]
    to_output_layer[to_row, tow_column] = 1

    return np.stack([from_output_layer, to_output_layer])

class ChessDataset(Dataset):
    def __init__(self, games,player_name):
        super(ChessDataset, self).__init__()
        self.games = games
        self.player_name = player_name
        self.games_number = self.games.shape[0]
        
    def __len__(self):
        return self.games_number
    
    def __getitem__(self, index):
        random_i = np.random.randint(0, self.games.shape[0]-2)
        start_i = random_i - self.games['move_no'].values[random_i] + 1
        flag = False
        if self.games['player'].values[random_i] == player_name:
            flag = True if self.games['color'].values[random_i] == 'Black' else False
            next_move = self.games['move'].values[random_i]
            moves = self.games['move'].values[start_i:random_i]
        else:
            flag = True if self.games['color'].values[random_i] == 'White' else False
            if int(self.games['move_no'].values[random_i+1]) > \
            int(self.games['move_no'].values[random_i]):
                next_move = self.games['move'].values[random_i+1]
                moves = self.games['move'].values[start_i:random_i+1]
            else:
                next_move = self.games['move'].values[random_i-1]
                moves = self.games['move'].values[start_i:random_i-1]
        board = chess.Board()
        for move in moves:
            board.push_uci(move)
        x = board_2_rep(board)
        y = move_2_rep(next_move, board)
        if flag:
            x*=-1
        return x, y

data_train = ChessDataset(df, player_name)
data_train_loader = DataLoader(data_train, batch_size = 32, shuffle=True, drop_last=True)

class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += x_input
        x = self.activation2(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x

def checkmate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()
    return None

def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs ** 3
    probs = probs / probs.sum()
    return probs

def predict(x):
    '''    
    Predicts the move probabilities using a neural network model.
        - Loads the pre-trained model.
        - Evaluates the model on the input data.
        - Returns the predicted move probabilities.
    '''
    model = ChessNet(hidden_layers=4, hidden_size=200).to(device)
    model.load_state_dict(torch.load(r'src/models/model.pth'))
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        return outputs.cpu().numpy()
    
def stockfish(board):
    '''
     Uses Stockfish to analyze the current board position.
        - Loads the Stockfish engine.
        - Analyzes the board position for a short time.
        - Returns the best move suggested by Stockfish and the evaluation score.
    '''
    import chess.engine
    from stockfish import Stockfish
    
    stockfish_path = r'' # your path to stockfish engine example: 'C:/Users/.../stockfish_13_win_x64_avx2/stockfish_13_win_x64_avx2.exe'
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine: 
        result = engine.analyse(board, chess.engine.Limit(time=0.1))
        
        #get best move 
        stockfish=Stockfish(stockfish_path)
        stockfish.set_fen_position(board.fen())
        suggested_move = stockfish.get_best_move()
        
        if suggested_move is None:
            return None, None

        evaluation = result['score']
        numeric_part = evaluation.relative.score() #if > 0, white's better.
        if type(numeric_part) == type(None):
                numeric_part = 0
    
        return numeric_part, suggested_move

def choose_move(board: chess.Board, color) -> chess.Move:
    legal_moves = list(board.legal_moves)
    
    move = checkmate_single(board)

    if move is not None:
        return move
    
    x = torch.Tensor(board_2_rep(board)).float().to(device)
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0)
    move = predict(x)
    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))
    for from_ in froms:
        # print(move[0,:,:][0][0])
        val = move[0,:,:][0][8-int(from_[1]), letter_2_num[from_[0]]]
        # print(from_)
        vals.append(val)
    
    probs = distribution_over_moves(vals)

    chosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        if from_ == chosen_from:
            to = str(legal_move)[2:]
            # print(move[0,:,:][0])
            # print(move[0,:,:][1])
            val = move[0,:,:][1][8 - int(to[1]), letter_2_num[to[0]]]
            vals.append(val)
        else:
            vals.append(0)
    chosen_move = legal_moves[np.argmax(vals)]
    
    #stockfish evaluates AI classifier move choice
    board2 = board.copy()
    #get evaluation before and after to check if blunder:
    evaluation_before, engine_move = stockfish(board)
    if type(evaluation_before) == type(None):
        evaluation_before = 0
        
    board2.push(chosen_move)
    evaluation_after, _ = stockfish(board2) #gets evaluation from opponent's perspective            
        
    if type(evaluation_after) == type(None):
        evaluation_after = 0

    dif_eval = abs(evaluation_before - evaluation_after)
    print(dif_eval)
    random_number = random.uniform(0, 1)
    
    if dif_eval > 200 and evaluation_before > evaluation_after: #if blunder
        if random_number > 0.05:
            if engine_move is None:
                return chosen_move
            else:
                return engine_move #correct with engine move
    else:           
        return chosen_move

model_path = "\\src\models\model.pth"
file_model = [f for f in os.listdir(current_directory)]

if 'model.pth' not in model_path:    
    model = ChessNet(hidden_layers=4, hidden_size=200).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    record = []
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_train_loader):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)  # convert labels to float
            optimizer.zero_grad()
            outputs = model(inputs)
            output_from = outputs[:, 0, :]
            output_to = outputs[:, 1, :]
            y_from = labels[:, 0, :]
            y_to = labels[:, 1, :]
            loss_from = nn.CrossEntropyLoss()(output_from, y_from.argmax(dim=1))
            loss_to = nn.CrossEntropyLoss()(output_to, y_to.argmax(dim=1))
            loss = loss_from + loss_to
            loss.backward()
            optimizer.step()
            record.append(loss.item())
            if i % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(data_train_loader), loss.item()))
        torch.save(model.state_dict(), 'model.pth')


