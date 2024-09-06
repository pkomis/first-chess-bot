from PyQt5.QtSvg import QSvgWidget
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import pyqtSlot
from src.logic_chess import *
import chess
import chess.svg
import random
from src.Ai import choose_move
from PyQt5.QtWidgets import *
            
class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Chess bot"
        self.color = False
        self.position_from = ""
        self.position_to = ""
        self.setLayout(qtw.QVBoxLayout())
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 880, 880)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 800, 800)
        self.chessboard = chess.Board()
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg) 
        self.layout().addWidget(self.widgetSvg)        
        
        self.gridLayoutWidget = qtw.QWidget(self)
        self.gridLayoutWidget.setGeometry(40, 40, 800, 770)
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = qtw.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.button()
        
        self.textbox = qtw.QLineEdit(self)
        self.textbox.returnPressed.connect(self.on_click)
        self.layout().addWidget(self.textbox)
        
        self.show()
    
    def button_clicked(self):
        # Get the sender of the signal
        button = self.sender()
        self.count = 0
        # Find the position of the button
        for i in range(8):
            for j in range(8):
                if self.buttons[i][j] == button:
                    if not self.color:
                        # Convert the position to chess notation (e.g., 'e4')
                        self.position = chr(97 + j) + str(8 - i)
                        if not self.position_from:
                            self.position_from = self.position
                        elif not self.position_to and self.position_from != self.position:
                            self.position_to = self.position
                            self.player_move()
                            
                        elif self.position_from == self.position:
                            self.position_from = ""
                            self.position_to = ""
                        else:
                            self.position_from = self.position
                            self.position_to = ""
                        
                    else:
                        self.position = chr(97 + 7 - j) + str(i + 1)                        
                        if not self.position_from:
                            self.position_from = self.position
                        elif not self.position_to and self.position_from != self.position:
                            self.position_to = self.position
                            self.player_move()
                        elif self.position_from == self.position:
                            self.position_from = ""
                            self.position_to = ""
                        else:
                            self.position_from = self.position
                            self.position_to = ""
                        
    def button(self):
        self.buttons = [[QPushButton() for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                button = self.buttons[i][j]
                button.setFixedSize(91, 91)  # Set button size
                #button.setContentsMargins(10, 10, 10, 10)
                button.clicked.connect(self.button_clicked)  # Connect to slot
                button.setStyleSheet("background:transparent;")
                self.gridLayout.addWidget(button, i, j)  # Add button to layout
    
    def show_board(self):
        if self.color:
            self.count = self.count
        else:
            self.count = 64 - self.count
        self.chessboardSvg = chess.svg.board(
            self.chessboard,
            flipped=self.color,
            lastmove = self.move if self.move != None else None,
            fill = dict.fromkeys(chess.SquareSet([self.count]), "#FFFF00"),
            ).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
        self.count = 0
        
    def bot(self):
        '''
            Executes the bot's move.
            - Chooses a move for the bot using the `choose_move` function.
            - Pushes the move to the chessboard.
            - Updates the chessboard SVG with the new move.
       '''
        move = str(choose_move(self.chessboard, self.chessboard.turn))
        self.chessboard.push_san(move)
        self.chessboardSvg = chess.svg.board(
            self.chessboard, 
            lastmove=chess.Move.from_uci(move), 
            flipped=self.color).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
        
        if self.chessboard.is_checkmate() or draw(self.chessboard):
            self.show_endofgame_messagebox()
            
    def random_game(self):
        # Randomly decide the color for the bot and start the game
        if random.randint(0,1):
            self.color = True
            self.bot()
            
    @pyqtSlot()
    def on_click(self):
        # Get the player's move from the textbox
        move = self.textbox.text() # pobranie ruchu gracza

        # Check if the move is legal and the game is not over
        if (islegal_move(move, self.chessboard) and
            not self.chessboard.is_checkmate() and 
            not draw(self.chessboard)):
            
             # Push the player's move to the chessboard
            self.chessboard.push_uci(move)
            
            # Update the chessboard SVG with the new move
            self.chessboardSvg = chess.svg.board(
                self.chessboard,
                lastmove=chess.Move.from_uci(move), 
                flipped=self.color).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
            
            if self.chessboard.is_checkmate() or draw(self.chessboard):
                self.show_endofgame_messagebox()
            else:
                self.bot()
        else:
            self.show_warning_messagebox()    
        self.textbox.setText("") # wyczyszczenie pola tekstowego
            
    def player_move(self):
        if self.position_from and self.position_to:
            move = self.position_from + self.position_to
            if (islegal_move(move, self.chessboard) and
                not self.chessboard.is_checkmate() and 
                not draw(self.chessboard)):
                self.chessboard.push_uci(move)
                self.chessboardSvg = chess.svg.board(
                    self.chessboard,
                    lastmove=chess.Move.from_uci(move), 
                    flipped=self.color).encode("UTF-8")
                self.widgetSvg.load(self.chessboardSvg)
                
                if self.chessboard.is_checkmate() or draw(self.chessboard):
                    self.show_endofgame_messagebox()
                else:
                    self.bot()
            else:
                self.show_warning_messagebox()
            self.position_from = None
            self.position_to = None

    def show_warning_messagebox(self): 
        self.msg = QMessageBox() 
        self.msg.setIcon(QMessageBox.Warning) 
        self.msg.setText("Niepoprawny ruch. Spróbuj ponownie.") 
        self.msg.setWindowTitle("Warning") 
        self.msg.setStandardButtons(QMessageBox.Ok) 

        self.retval = self.msg.exec_()

    def show_endofgame_messagebox(self): 
        self.msg = QMessageBox() 
        self.msg.setIcon(QMessageBox.Information) 
        self.msg.setText(f"Wynik partii: {self.chessboard.outcome().result()}") 
        self.msg.setWindowTitle("Information") 
        self.msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel) 

        self.retval = self.msg.exec_()
            
if __name__ == "__main__":
    
    app = qtw.QApplication([])
    board = chess.Board()
    win = MainWindow()
    chessboardSvg = chess.svg.board(board).encode("UTF-8")
    
    win.widgetSvg.load(chessboardSvg)
    win.random_game()
    app.processEvents() 
    app.exec_()
