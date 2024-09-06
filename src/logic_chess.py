from chess import Board


def islegal_move(player_move, board:Board):
    legal_moves = [move.uci() for move in board.legal_moves]
    try:
        if player_move in legal_moves:
            return True
        
    except ValueError or player_move not in legal_moves:
        return False
    
def ischeckmate(board:Board):
    if board.is_checkmate():
        return True
    else: 
        return False
    
def draw(board:Board):
    if board.is_stalemate():
        print("Draw by the stalemate")
        return True
    elif board.is_insufficient_material():
        print("Draw by the insufficient material")
        return True
    elif board.is_seventyfive_moves():
        print("Draw by the seventyfive moves")
        return True
    elif board.is_fivefold_repetition():
        print("Draw by the fivefold repetition")
        return True
    else:
        return False