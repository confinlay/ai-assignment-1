import numpy as np
import random
from typing import Tuple, List
from collections import deque

class Maze:
    """
    A class to generate mazes using the Recursive Backtracking algorithm.
    The maze is represented as a 2D list where:
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
        
        # Define entrance and exit positions
        self.entrance = (1, 0)
        self.exit = (self.height-2, self.width-1)
    
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
        self.maze[self.entrance[0]][self.entrance[1]] = 0  # Entrance
        self.maze[self.exit[0]][self.exit[1]] = 0  # Exit
        
        return self.maze
    
    def generate_non_perfect(self, removal_percentage: float = 0.1) -> List[List[int]]:
        """
        Generate a non-perfect maze with multiple paths to the exit.
        Simply removes a percentage of internal walls from a perfect maze.
        
        Args:
            removal_percentage (float): Percentage of internal walls to remove (0.0 to 1.0)
            
        Returns:
            The modified maze with multiple solution paths
        """
        # First generate a perfect maze
        self.generate()
        
        # Find internal walls (walls with paths on both sides)
        internal_walls = []
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                # Skip non-wall cells
                if self.maze[y][x] != 1:
                    continue
                
                # Check if this wall has paths on opposite sides (horizontally or vertically)
                if ((y > 0 and y < self.height-1 and self.maze[y-1][x] == 0 and self.maze[y+1][x] == 0) or
                    (x > 0 and x < self.width-1 and self.maze[y][x-1] == 0 and self.maze[y][x+1] == 0)):
                    internal_walls.append((y, x))
        
        # Determine how many walls to remove (at least 1)
        num_walls_to_remove = max(1, int(len(internal_walls) * removal_percentage))
        
        # Randomly select and remove walls
        if internal_walls:
            walls_to_remove = random.sample(internal_walls, min(num_walls_to_remove, len(internal_walls)))
            for y, x in walls_to_remove:
                self.maze[y][x] = 0
        
        return self.maze
    
    def _find_shortest_path(self) -> List[Tuple[int, int]]:
        """
        Find the shortest path from entrance to exit using BFS.
        
        Returns:
            List of (y, x) coordinates representing the shortest path
        """
        # BFS uses a queue to find the shortest path
        queue = deque([(self.entrance, [self.entrance])])
        visited = {self.entrance}
        
        while queue:
            (y, x), path = queue.popleft()
            
            # Check if we've reached the exit
            if (y, x) == self.exit:
                return path
            
            # Check all four directions
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                
                # Check if the neighbor is valid and not visited
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.maze[ny][nx] == 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    new_path = path + [(ny, nx)]
                    queue.append(((ny, nx), new_path))
        
        # If no path is found, return an empty list
        return []
    
    def print_maze(self):
        """Print a visual representation of the maze."""
        for row in self.maze:
            print(''.join(['██' if cell == 1 else '  ' for cell in row]))

    def show_maze(self, highlight_path=None):
        """
        Display the maze in a popup window using matplotlib.
        
        Args:
            highlight_path: List of (y, x) coordinates to highlight (e.g., solution path)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create a figure with appropriate size
            plt.figure(figsize=(10, 10))
            
            # Convert maze to numpy array for display
            maze_array = np.array(self.maze)
            
            # Create a colored version for display
            colored_maze = np.zeros((self.height, self.width, 3))
            colored_maze[maze_array == 1] = [0, 0, 0]  # Walls are black
            colored_maze[maze_array == 0] = [1, 1, 1]  # Paths are white
            
            # Highlight entrance and exit
            colored_maze[self.entrance[0], self.entrance[1]] = [0, 1, 0]  # Green entrance
            colored_maze[self.exit[0], self.exit[1]] = [1, 0, 0]  # Red exit
            
            # Highlight path if provided
            if highlight_path:
                for y, x in highlight_path:
                    # Skip entrance and exit to keep their colors
                    if (y, x) != self.entrance and (y, x) != self.exit:
                        colored_maze[y, x] = [0, 0, 1]  # Blue path
            
            # Display the maze
            plt.imshow(colored_maze)
            plt.axis('off')
            plt.title(f"Maze ({self.width}x{self.height})")
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required to show the maze.")
            print("Install it with: pip install matplotlib")

    def analyze_maze(self):
        """
        Simple analysis of the maze - just find the shortest path length.
        
        Returns:
            Dictionary with analysis results
        """
        # Find the shortest path
        shortest_path = self._find_shortest_path()
        
        return {
            "shortest_path_length": len(shortest_path),
            "is_perfect": None,  # We no longer try to detect this
            "number_of_paths": "unknown"  # We no longer try to count paths
        }


# Example usage
if __name__ == "__main__":
    # Create a maze generator instance
    size = 21
    my_maze = Maze(size, size)
    
    # Generate a perfect maze
    print(f"Generating a {size}x{size} perfect maze...")
    maze = my_maze.generate()
    
    # Find and print the shortest path
    path = my_maze._find_shortest_path()
    print(f"Perfect maze - shortest path length: {len(path)}")
    
    # Show the maze with solution
    my_maze.show_maze(highlight_path=path)
    
    # Generate a non-perfect maze
    print(f"Generating a {size}x{size} non-perfect maze...")
    maze = my_maze.generate_non_perfect(removal_percentage=0.15)
    
    # Find and show the shortest path
    path = my_maze._find_shortest_path()
    print(f"Non-perfect maze - shortest path length: {len(path)}")
    my_maze.show_maze(highlight_path=path)