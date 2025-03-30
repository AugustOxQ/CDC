import random

import pygame

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
GRID_SIZE = 30
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Tetromino shapes
TETROMINOS = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
}

# Colors for each tetromino
COLORS = {
    "I": CYAN,
    "O": YELLOW,
    "T": MAGENTA,
    "S": GREEN,
    "Z": RED,
    "J": BLUE,
    "L": ORANGE,
}

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

# Clock to control the frame rate
clock = pygame.time.Clock()


# Function to draw the grid
def draw_grid():
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)


# Function to draw the tetromino
def draw_tetromino(tetromino, color, offset_x, offset_y):
    for y, row in enumerate(tetromino):
        for x, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    (x + offset_x) * GRID_SIZE,
                    (y + offset_y) * GRID_SIZE,
                    GRID_SIZE,
                    GRID_SIZE,
                )
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)


# Function to check if a position is valid
def is_valid_position(tetromino, offset_x, offset_y):
    for y, row in enumerate(tetromino):
        for x, cell in enumerate(row):
            if cell:
                new_x = x + offset_x
                new_y = y + offset_y
                if (
                    new_x < 0
                    or new_x >= GRID_WIDTH
                    or new_y >= GRID_HEIGHT
                    or (new_y >= 0 and grid[new_y][new_x])
                ):
                    return False
    return True


# Function to rotate a tetromino
def rotate_tetromino(tetromino):
    return [list(row) for row in zip(*tetromino[::-1])]


# Function to clear full lines
def clear_lines():
    global score
    full_lines = [i for i, row in enumerate(grid) if all(row)]
    for line in full_lines:
        del grid[line]
        grid.insert(0, [0] * GRID_WIDTH)
        score += 100


# Main game loop
def main():
    global grid, score
    grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    score = 0
    current_tetromino = random.choice(list(TETROMINOS.keys()))
    current_color = COLORS[current_tetromino]
    current_tetromino = TETROMINOS[current_tetromino]
    offset_x = GRID_WIDTH // 2 - len(current_tetromino[0]) // 2
    offset_y = 0
    game_over = False
    last_fall_time = pygame.time.get_ticks()

    while not game_over:
        screen.fill(BLACK)
        draw_grid()
        draw_tetromino(current_tetromino, current_color, offset_x, offset_y)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if is_valid_position(current_tetromino, offset_x - 1, offset_y):
                        offset_x -= 1
                elif event.key == pygame.K_RIGHT:
                    if is_valid_position(current_tetromino, offset_x + 1, offset_y):
                        offset_x += 1
                elif event.key == pygame.K_DOWN:
                    if is_valid_position(current_tetromino, offset_x, offset_y + 1):
                        offset_y += 1
                elif event.key == pygame.K_UP:
                    rotated_tetromino = rotate_tetromino(current_tetromino)
                    if is_valid_position(rotated_tetromino, offset_x, offset_y):
                        current_tetromino = rotated_tetromino

        current_time = pygame.time.get_ticks()
        if current_time - last_fall_time > 500:
            if is_valid_position(current_tetromino, offset_x, offset_y + 1):
                offset_y += 1
            else:
                for y, row in enumerate(current_tetromino):
                    for x, cell in enumerate(row):
                        if cell:
                            grid[offset_y + y][offset_x + x] = 1
                clear_lines()
                current_tetromino = random.choice(list(TETROMINOS.keys()))
                current_color = COLORS[current_tetromino]
                current_tetromino = TETROMINOS[current_tetromino]
                offset_x = GRID_WIDTH // 2 - len(current_tetromino[0]) // 2
                offset_y = 0
                if not is_valid_position(current_tetromino, offset_x, offset_y):
                    game_over = True
            last_fall_time = current_time

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
