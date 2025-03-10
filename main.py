import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = HEIGHT // ROWS

# Colors
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
BLUE, RED = (0, 0, 255), (255, 0, 0)
HIGHLIGHT = (0, 255, 0)  # Yellow for highlighting moves

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers")

# Game state
current_player = "blue"
selected_piece = None
forced_jumps = []
additional_turn = False
first_move = True  # Track if the current move is the first one

class Piece:
    def __init__(self, color, rank, row, col):
        self.color = color
        self.rank = rank  # "pawn" or "king"
        self.row = row
        self.col = col
    
    def promote(self):
        """ Promote the piece to a king. """
        self.rank = "king"
    
    def move(self, new_row, new_col):
        """ Move the piece to a new position. """
        self.row = new_row
        self.col = new_col


def init_board():
    """ Initialize the board with pieces. """
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]
    for row in range(ROWS):
        for col in range(COLS):
            if (row + col) % 2 == 1:  # Place pieces only on black squares
                if row < 3:  # Top 3 rows for blue
                    board[row][col] = Piece("blue", "pawn", row, col)
                elif row >= ROWS - 3:  # Bottom 3 rows for red
                    board[row][col] = Piece("red", "pawn", row, col)
    return board


def valid_move(start, end, board):
    """ Check if a move is valid, including jumps. """
    sr, sc = start
    er, ec = end
    piece = board[sr][sc]

    if piece is None:
        return False

    color, rank = piece.color, piece.rank

    # Normal move (one square diagonal)
    if abs(sr - er) == 1 and abs(sc - ec) == 1 and board[er][ec] is None:
        if rank == "pawn":
            if (color == "blue" and er > sr) or (color == "red" and er < sr):  # Forward only
                return True
        elif rank == "king":
            return True  # Kings can move in both directions

    # Jump move (two squares diagonal)
    elif abs(sr - er) == 2 and abs(sc - ec) == 2:
        mid_r, mid_c = (sr + er) // 2, (sc + ec) // 2
        if board[mid_r][mid_c] is not None and board[mid_r][mid_c].color != color and board[er][ec] is None:
            if rank == "pawn":  # Restrict jumps based on pawn movement rules
                if (color == "blue" and er > sr) or (color == "red" and er < sr):
                    return "jump"
            elif rank == "king":  # Kings can jump both ways
                return "jump"

    return False


def get_available_jumps(piece_pos, board):
    """ Get all possible jumps from a position for multiple jump logic. """
    sr, sc = piece_pos
    piece = board[sr][sc]
    if piece is None:
        return []

    color, rank = piece.color, piece.rank
    directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    jumps = []

    for dr, dc in directions:
        er, ec = sr + dr, sc + dc
        if 0 <= er < ROWS and 0 <= ec < COLS and valid_move((sr, sc), (er, ec), board) == "jump":
            jumps.append((er, ec))

    return jumps

def make_move(start, end, board):
    """ Move the piece and handle captures/kings. """
    global current_player, forced_jumps, additional_turn, selected_piece, first_move

    sr, sc = start
    er, ec = end
    piece = board[sr][sc]

    is_jump = valid_move(start, end, board) == "jump"
    print(f"Making move from ({sr}, {sc}) to ({er}, {ec})")
    
    if is_jump:
        mid_r, mid_c = (sr + er) // 2, (sc + ec) // 2
        board[mid_r][mid_c] = None  # Remove jumped piece

    piece.move(er, ec)
    board[er][ec] = piece
    board[sr][sc] = None

    # King promotion
    if (piece.color == "blue" and er == ROWS - 1) or (piece.color == "red" and er == 0):
        piece.promote()

    # Check for more jumps from new position
    forced_jumps = get_available_jumps((er, ec), board)
    print(f"Forced jumps available: {forced_jumps}")

    if is_jump and forced_jumps:
        additional_turn = True
        selected_piece = (er, ec)  # Keep selected piece for more jumps
        print(f"Bonus turn, keeping piece at {selected_piece}")
    else:
        additional_turn = False
        selected_piece = None  # Reset after move
        current_player = "red" if current_player == "blue" else "blue"  # Switch turns
        print(f"Switching turn to {current_player}")

    first_move = False  # Mark that the first move has been completed


def draw_board(board):
    """ Draw the board with pieces and highlights. """
    for r in range(ROWS):
        for c in range(COLS):
            pygame.draw.rect(screen, WHITE if (r + c) % 2 == 0 else BLACK, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] is not None:
                piece = board[r][c]
                color = BLUE if piece.color == "blue" else RED
                pygame.draw.circle(screen, color, (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)
                font = pygame.font.Font(None, 40)
                text = font.render("K" if piece.rank == "king" else "P", True, WHITE)
                screen.blit(text, text.get_rect(center=(c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)))


def highlight_moves(selected_piece, board):
    """ Highlight valid normal moves and jumps. """
    if selected_piece:
        sr, sc = selected_piece
        piece = board[sr][sc]

        jumps = get_available_jumps((sr, sc), board)
        normal_moves = []

        # Check normal moves
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            er, ec = sr + dr, sc + dc
            if 0 <= er < ROWS and 0 <= ec < COLS and valid_move((sr, sc), (er, ec), board):
                normal_moves.append((er, ec))

        # Highlight jumps first
        for er, ec in jumps:
            pygame.draw.circle(screen, HIGHLIGHT, 
                               (ec * SQUARE_SIZE + SQUARE_SIZE // 2, er * SQUARE_SIZE + SQUARE_SIZE // 2), 
                               SQUARE_SIZE // 3)

        # Highlight normal moves even when a jump exists
        for er, ec in normal_moves:
            pygame.draw.circle(screen, HIGHLIGHT, 
                               (ec * SQUARE_SIZE + SQUARE_SIZE // 2, er * SQUARE_SIZE + SQUARE_SIZE // 2), 
                               SQUARE_SIZE // 3)


def handle_events(board):
    """ Handle user input events. """
    global selected_piece, additional_turn, current_player, first_move

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            col, row = x // SQUARE_SIZE, y // SQUARE_SIZE

            if event.button == 1:  # Left click
                if additional_turn:
                    # During a bonus turn, continue with the selected piece and allow jumps
                    if selected_piece == (row, col):
                        pass  # Do nothing if clicking the same piece
                    else:
                        # If the clicked position is not the same as selected, check if it's a valid jump
                        if valid_move(selected_piece, (row, col), board):
                            make_move(selected_piece, (row, col), board)
                            # Keep the piece selected if there are more jumps
                            if additional_turn:
                                selected_piece = (row, col)
                            else:
                                selected_piece = None  # Reset after move
                else:
                    if board[row][col] is not None and board[row][col].color == current_player:
                        selected_piece = (row, col)  # Select the piece
                    elif selected_piece is not None:  # Attempting to move the selected piece
                        if valid_move(selected_piece, (row, col), board):
                            make_move(selected_piece, (row, col), board)
                            # If the move was a jump and additional turn is active, keep the piece selected
                            if additional_turn:
                                selected_piece = (row, col)
                            else:
                                selected_piece = None  # Reset after move
            elif event.button == 3:  # Right-click to skip bonus turn
                if additional_turn:
                    additional_turn = False
                    selected_piece = None
                    current_player = "red" if current_player == "blue" else "blue"


# Game loop
board = init_board()
while True:
    handle_events(board)
    draw_board(board)
    highlight_moves(selected_piece, board)
    pygame.display.flip()
