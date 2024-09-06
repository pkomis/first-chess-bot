from converter.pgn_data import PGNData
import os

def check_if_csv_exists():
    current_directory = os.path.join(os.path.dirname(__file__), 'data')
    file_chess_move = [f for f in os.listdir(current_directory) 
                       if f.endswith(f".csv")]
    if file_chess_move is None:
        return False
    else:
        return True

current_directory = os.path.join(os.path.dirname(__file__), 'data')
file_paths = [f for f in os.listdir(current_directory) if f.endswith(f".pgn")]
files = os.listdir(current_directory)


if check_if_csv_exists() == True or len(file_paths) > 0:
    black = os.path.join(current_directory, file_paths[0])
    white = os.path.join(current_directory, file_paths[1])
    pgn_data = PGNData([white, black])
    pgn_data.export()
    print("The files have been converted to csv.")
elif len(file_paths) < 0:
    print("There are not enough files to convert to csv.")
    print("Please provide two pgn files.")
    print("Exiting the program.")
    exit(1)
