# Chess AI Project

This project is a Chess AI application that uses a combination of a neural network and the Stockfish chess engine to play chess. The application allows users to play against the AI, which can make moves based on its evaluation of the board state.

## Features

- **Play against AI**: Users can play chess against the AI.
- **Random Game Initialization**: The game can start with a random color for the bot.
- **Move Prediction**: The AI predicts the best move using a neural network model.
- **Stockfish Integration**: The AI uses the Stockfish engine for move analysis and evaluation.
- **Interactive Chessboard**: The chessboard updates in real-time with the moves made by the player and the AI.

## Project Structure
project_root/ │ ├── src/ │ ├── data/ │ │ └── your_csv_files.csv │ ├── models/ │ │ └── model.pth │ ├── main.py │ ├── ai.py │ └── ... ├── README.md └── ...


## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/chess-ai-project.git
    cd chess-ai-project
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Stockfish**:
    - Download the Stockfish engine from [Stockfish's official website](https://stockfishchess.org/download/).
    - Place the Stockfish executable in the appropriate directory (e.g., `C:/Users/.../stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe`).

## Usage

1. **Run the application**:
    ```bash
    python src/main.py
    ```

2. **Play the game**:
    - The game will start with a random color for the bot.
    - Enter your moves in the textbox and click the button to make a move.
    - The AI will respond with its move, and the chessboard will update accordingly.

## Acknowledgements

- [Stockfish Chess Engine](https://stockfishchess.org/)
- [Python-Chess Library](https://python-chess.readthedocs.io/en/latest/)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
