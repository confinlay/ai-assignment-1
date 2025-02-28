import numpy as np
import random
from typing import Tuple, List

class Maze:
    """
    A class to generate perfect mazes using the Recursive Backtracking algorithm.
    The maze is represented as a 2D numpy array where:
    - 1 represents walls
    - 0 represents paths
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize the maze with given dimensions.
        
        Args:
            width (int): The width of the maze (must be odd)
            height (int): The height of the maze (must be odd)
        """
        # Ensure dimensions are odd
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        
        # Initialize maze with all walls
        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        # Define possible directions: (y, x) - right, down, left, up
        self.directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    
    def generate(self) -> List[List[int]]:
        """Generate a perfect maze using an iterative approach."""
        # Start from cell (1,1)
        start_y, start_x = 1, 1
        self.maze[start_y][start_x] = 0
        
        # Use a stack to track cells to visit
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack[-1]  # Get current cell (top of stack)
            
            # Randomize directions
            directions = self.directions.copy()
            random.shuffle(directions)
            
            # Find any unvisited neighbors
            found_neighbor = False
            
            for dy, dx in directions:
                new_y, new_x = y + dy, x + dx
                
                # Check if the new position is valid and unvisited
                if (0 < new_y < self.height-1 and 
                    0 < new_x < self.width-1 and 
                    self.maze[new_y][new_x] == 1):
                    
                    # Carve a path
                    self.maze[new_y][new_x] = 0
                    self.maze[y + dy//2][x + dx//2] = 0
                    
                    # Add new cell to stack
                    stack.append((new_y, new_x))
                    found_neighbor = True
                    break
            
            # If no unvisited neighbors, backtrack
            if not found_neighbor:
                stack.pop()
        
        # Create entrance and exit
        self.maze[1][0] = 0  # Entrance
        self.maze[self.height-2][self.width-1] = 0  # Exit
        
        return self.maze
    
    def print_maze(self):
        """Print a visual representation of the maze."""
        for row in self.maze:
            print(''.join(['██' if cell == 1 else '  ' for cell in row]))

    def save_maze_as_image(self, filename="maze.png"):
        """
        Save the maze as a PNG image file.
        
        Args:
            filename (str): The name of the output file
        """
        try:
            from PIL import Image
            
            # Create a new black and white image
            img = Image.new('1', (self.width, self.height), 1)
            pixels = img.load()
            
            # Fill in the maze (0=white path, 1=black wall)
            for y in range(self.height):
                for x in range(self.width):
                    # Invert colors: 0 (path) becomes white (1), 1 (wall) becomes black (0)
                    pixels[x, y] = 1 - self.maze[y][x]
            
            # Save the image
            img.save(filename)
            print(f"Maze saved as {filename}")
            
        except ImportError:
            print("PIL (Pillow) library is required to save images.")
            print("Install it with: pip install pillow")

    def show_maze(self):
        """
        Display the maze in a popup window using matplotlib.
        The window will stay open until closed by the user.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create a figure with appropriate size
            plt.figure(figsize=(10, 10))
            
            # Display the maze (0=white path, 1=black wall)
            plt.imshow(self.maze, cmap='binary')
            
            # Remove axes and ticks for cleaner display
            plt.axis('off')
            
            # Add a title
            plt.title(f"Maze ({self.width}x{self.height})")
            
            # Show the plot in a non-blocking way
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required to show the maze.")
            print("Install it with: pip install matplotlib")

# Example usage
if __name__ == "__main__":
    # Create a maze generator instance
    size = 201
    my_maze = Maze(size, size)
    
    # Generate the maze
    print(f"Generating {size}x{size} maze...")
    maze = my_maze.generate()
    
    # Show the maze in a popup window
    my_maze.show_maze()