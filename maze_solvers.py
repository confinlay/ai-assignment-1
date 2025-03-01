from typing import List, Tuple, Dict, Set, Callable, Optional
from collections import deque
import heapq
import time
import numpy as np
from abc import ABC, abstractmethod

class BaseMazeSolver(ABC):
    """
    Abstract base class for maze solving algorithms.
    The maze is represented as a 2D list where:
    - 1 represents walls
    - 0 represents paths
    """
    
    def __init__(self, maze: List[List[int]]):
        """
        Initialize the maze solver with a maze.
        
        Args:
            maze: 2D list representing the maze (1=wall, 0=path)
        """
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)
        # Start position is always (1,0) in our maze
        self.start = (1, 0)
        # End position is always (height-2, width-1) in our maze
        self.end = (self.height-2, self.width-1)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns valid neighboring positions (up, right, down, left).
        
        Args:
            pos: Current position as (row, col) tuple
            
        Returns:
            List of valid neighboring positions
        """
        x, y = pos
        # Check all four directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        neighbors = []
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            # Check if the neighbor is within bounds and is a path (0)
            if (0 <= new_x < self.height and 
                0 <= new_y < self.width and 
                self.maze[new_x][new_y] == 0):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    @abstractmethod
    def solve(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Abstract method to be implemented by concrete solver classes.
        
        Returns:
            Tuple containing:
            - The path from start to end (if found)
            - Dictionary with metrics (nodes explored, time taken, path length)
            - Set of all visited nodes
        """
        pass


class DFSMazeSolver(BaseMazeSolver):
    """Depth-First Search implementation for maze solving."""
    
    def solve(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Solve the maze using Depth-First Search.
        
        Returns:
            Tuple containing:
            - The path from start to end (if found)
            - Dictionary with metrics (nodes explored, time taken, path length)
            - Set of all visited nodes
        """
        start_time = time.time()
        stack = [(self.start, [self.start])]
        visited = {self.start}
        nodes_explored = 0
        
        while stack:
            current, path = stack.pop()
            nodes_explored += 1
            
            if current == self.end:
                end_time = time.time()
                metrics = {
                    'nodes_explored': nodes_explored,
                    'time_taken': end_time - start_time,
                    'path_length': len(path)
                }
                return path, metrics, visited
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
        
        end_time = time.time()
        metrics = {
            'nodes_explored': nodes_explored,
            'time_taken': end_time - start_time,
            'path_length': 0  # No path found
        }
        return [], metrics, visited  # No path found


class BFSMazeSolver(BaseMazeSolver):
    """Breadth-First Search implementation for maze solving."""
    
    def solve(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Solve the maze using Breadth-First Search.
        
        Returns:
            Tuple containing:
            - The path from start to end (if found)
            - Dictionary with metrics (nodes explored, time taken, path length)
            - Set of all visited nodes
        """
        start_time = time.time()
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        nodes_explored = 0
        
        while queue:
            current, path = queue.popleft()
            nodes_explored += 1
            
            if current == self.end:
                end_time = time.time()
                metrics = {
                    'nodes_explored': nodes_explored,
                    'time_taken': end_time - start_time,
                    'path_length': len(path)
                }
                return path, metrics, visited
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        end_time = time.time()
        metrics = {
            'nodes_explored': nodes_explored,
            'time_taken': end_time - start_time,
            'path_length': 0  # No path found
        }
        return [], metrics, visited  # No path found


class AStarMazeSolver(BaseMazeSolver):
    """A* Search implementation for maze solving with configurable heuristics."""
    
    def manhattan_distance(self, pos: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            pos: Current position
            
        Returns:
            Manhattan distance to the goal
        """
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])
    
    def euclidean_distance(self, pos: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance heuristic (straight-line distance).
        
        Args:
            pos: Current position
            
        Returns:
            Euclidean distance to the goal
        """
        return ((pos[0] - self.end[0]) ** 2 + (pos[1] - self.end[1]) ** 2) ** 0.5
    
    def diagonal_distance(self, pos: Tuple[int, int]) -> float:
        """
        Calculate Chebyshev distance heuristic.
        This is the maximum of the horizontal and vertical distances.
        
        Args:
            pos: Current position
            
        Returns:
            Diagonal distance to the goal
        """
        return max(abs(pos[0] - self.end[0]), abs(pos[1] - self.end[1]))
    
    def solve(self, heuristic: str = 'manhattan') -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Solve the maze using A* Search with configurable heuristic.
        
        Args:
            heuristic: The heuristic to use - 'manhattan', 'euclidean', or 'diagonal'
            
        Returns:
            Tuple containing:
            - The path from start to end (if found)
            - Dictionary with metrics (nodes explored, time taken, path length)
            - Set of all visited nodes
        """
        # Select the appropriate heuristic function
        if heuristic == 'euclidean':
            h_func = self.euclidean_distance
        elif heuristic == 'diagonal':
            h_func = self.diagonal_distance
        else:  # Default to manhattan
            h_func = self.manhattan_distance
            
        start_time = time.time()
        nodes_explored = 0
        visited = set()
        
        # Priority queue entries are: (f_score, node_counter, current_pos, path)
        # The node_counter is used as a tiebreaker for equal f_scores
        start_h_score = h_func(self.start)
        counter = 0  # Counter for tiebreaking
        pq = [(start_h_score, counter, self.start, [self.start])]
        # Keep track of g_scores (path length to reach each position)
        g_scores = {self.start: 0}
        
        while pq:
            _, _, current, path = heapq.heappop(pq)
            nodes_explored += 1
            
            # If we've already processed this position, skip it
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == self.end:
                end_time = time.time()
                metrics = {
                    'nodes_explored': nodes_explored,
                    'time_taken': end_time - start_time,
                    'path_length': len(path)
                }
                return path, metrics, visited
            
            for neighbor in self.get_neighbors(current):
                # g_score is the length of the path to neighbor
                tentative_g_score = g_scores[current] + 1
                
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + h_func(neighbor)
                    counter += 1  # Increment counter for unique priority
                    heapq.heappush(pq, (f_score, counter, neighbor, path + [neighbor]))
        
        end_time = time.time()
        metrics = {
            'nodes_explored': nodes_explored,
            'time_taken': end_time - start_time,
            'path_length': 0  # No path found
        }
        return [], metrics, visited  # No path found


class MazeSolutionVisualizer:
    """Class for visualizing maze solutions."""
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]):
        """
        Initialize the visualizer with maze data.
        
        Args:
            maze: 2D list representing the maze
            start: Start position as (row, col)
            end: End position as (row, col)
        """
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])
        self.start = start
        self.end = end
    
    def visualize_path_terminal(self, path: List[Tuple[int, int]]):
        """
        Visualize the maze with the solved path in the terminal.
        Path cells are marked with '◈◈'
        
        Args:
            path: List of coordinates forming the solution path
        """
        path_set = set(path)
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in path_set:
                    print('◈◈', end='')
                elif self.maze[i][j] == 1:
                    print('██', end='')
                else:
                    print('  ', end='')
            print()

    def visualize_search(self, path: List[Tuple[int, int]], visited: Set[Tuple[int, int]], 
                         color: str = 'blue', title: str = 'Maze Solution', block: bool = False):
        """
        Visualize the search results using matplotlib.
        
        Args:
            path: List of coordinates forming the solution path
            visited: Set of all coordinates that were visited during search
            color: Base color for visualization (path will be this color, visited nodes a lighter shade)
            title: Title for the plot
            block: Whether to block execution until the figure is closed
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import numpy as np
            
            # Create a mask for walls - this is separate from our visualization array
            wall_mask = np.zeros((self.height, self.width), dtype=bool)
            for y in range(self.height):
                for x in range(self.width):
                    if self.maze[y][x] == 1:  # If it's a wall
                        wall_mask[y, x] = True
            
            # Create visualization array for paths/visited/solution
            viz_array = np.zeros((self.height, self.width), dtype=np.int8)
            
            # Mark visited cells as 1
            for y, x in visited:
                if 0 <= y < self.height and 0 <= x < self.width:
                    viz_array[y, x] = 1
            
            # Mark path cells as 2
            for y, x in path:
                if 0 <= y < self.height and 0 <= x < self.width:
                    viz_array[y, x] = 2
            
            # Create the figure
            plt.figure(figsize=(10, 10))
            
            # Create colormap for paths/visited/solution
            base_color = mcolors.to_rgb(color)
            light_color = tuple([min(1.0, c + 0.5) for c in base_color])
            path_colors = ['white', light_color, base_color]
            
            # Plot the maze - first the paths/visited/solution
            img = plt.imshow(viz_array, cmap=mcolors.ListedColormap(path_colors), 
                             vmin=0, vmax=2)
            
            # Now overlay walls as pure black
            # This is the key - we're creating a masked array where walls are explicitly set black
            plt.imshow(np.ones_like(viz_array), 
                       cmap=mcolors.ListedColormap(['black']),
                       alpha=wall_mask.astype(float))  # Only show black where wall_mask is True
            
            # Add start and end markers
            plt.plot(self.start[1], self.start[0], 'go', markersize=8)  # Start in green
            plt.plot(self.end[1], self.end[0], 'ro', markersize=8)      # End in red
            
            # Remove axes and ticks
            plt.axis('off')
            
            # Add title
            plt.title(title)
            
            # Add a custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='Path (unvisited)'),
                Patch(facecolor='black', edgecolor='black', label='Wall'),
                Patch(facecolor=light_color, edgecolor='black', label='Visited'),
                Patch(facecolor=base_color, edgecolor='black', label='Solution path'),
                Patch(facecolor='green', edgecolor='black', label='Start'),
                Patch(facecolor='red', edgecolor='black', label='End')
            ]
            plt.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.05), ncol=3)
            
            plt.tight_layout()
            plt.show(block=block)
            
        except ImportError:
            print("Matplotlib is required for visualization.")
            print("Install it with: pip install matplotlib")

    def visualize_all_searches(self, algorithm_results, colors, titles):
        """
        Visualize multiple search results in a single figure with subplots.
        
        Args:
            algorithm_results: List of tuples (path, metrics, visited) for each algorithm
            colors: List of colors for each algorithm
            titles: List of titles for each algorithm
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import numpy as np
            from matplotlib.patches import Patch
            
            # Create a mask for walls - this is separate from our visualization array
            wall_mask = np.zeros((self.height, self.width), dtype=bool)
            for y in range(self.height):
                for x in range(self.width):
                    if self.maze[y][x] == 1:  # If it's a wall
                        wall_mask[y, x] = True
            
            # Setup the figure
            num_plots = len(algorithm_results)
            rows = (num_plots + 2) // 3  # Calculate number of rows needed
            cols = min(3, num_plots)     # Max 3 columns
            
            fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
            if num_plots == 1:
                axs = [axs]  # Make it iterable for single plot
            else:
                axs = axs.flatten()  # Convert 2D array to 1D for easy iteration
            
            # Create plots for each algorithm
            for i, ((path, metrics, visited), color, title) in enumerate(zip(algorithm_results, colors, titles)):
                ax = axs[i]
                
                # Create visualization array for paths/visited/solution
                viz_array = np.zeros((self.height, self.width), dtype=np.int8)
                
                # Mark visited cells as 1
                for y, x in visited:
                    if 0 <= y < self.height and 0 <= x < self.width:
                        viz_array[y, x] = 1
                
                # Mark path cells as 2
                for y, x in path:
                    if 0 <= y < self.height and 0 <= x < self.width:
                        viz_array[y, x] = 2
                
                # Create colormap for paths/visited/solution
                base_color = mcolors.to_rgb(color)
                light_color = tuple([min(1.0, c + 0.5) for c in base_color])
                path_colors = ['white', light_color, base_color]
                
                # Plot the maze - first the paths/visited/solution
                img = ax.imshow(viz_array, cmap=mcolors.ListedColormap(path_colors), 
                               vmin=0, vmax=2)
                
                # Now overlay walls as pure black
                ax.imshow(np.ones_like(viz_array), 
                         cmap=mcolors.ListedColormap(['black']),
                         alpha=wall_mask.astype(float))  # Only show black where wall_mask is True
                
                # Add start and end markers
                ax.plot(self.start[1], self.start[0], 'go', markersize=8)
                ax.plot(self.end[1], self.end[0], 'ro', markersize=8)
                
                # Remove axes and ticks
                ax.axis('off')
                
                # Add title with metrics
                path_length = len(path) if path else 0
                nodes_explored = metrics.get('nodes_explored', len(visited))
                ax.set_title(f"{title}\nPath Length: {path_length}\nNodes Explored: {nodes_explored}")
            
            # Hide any unused subplots
            for j in range(i+1, len(axs)):
                axs[j].axis('off')
                axs[j].set_visible(False)
            
            # Add a common legend at the bottom
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='Path (unvisited)'),
                Patch(facecolor='black', edgecolor='black', label='Wall'),
                Patch(facecolor='lightblue', edgecolor='black', label='Visited (generic)'),
                Patch(facecolor='blue', edgecolor='black', label='Solution path (generic)'),
                Patch(facecolor='green', edgecolor='black', label='Start'),
                Patch(facecolor='red', edgecolor='black', label='End')
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.01))
            
            plt.suptitle("Maze Solutions by Different Algorithms", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)  # Make room for the legend
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for visualization.")
            print("Install it with: pip install matplotlib")


class MazeSolver:
    """
    A facade class that provides a unified interface to all maze solving algorithms.
    This maintains backward compatibility with the original code.
    """
    
    def __init__(self, maze: List[List[int]]):
        """
        Initialize the maze solver with a maze.
        
        Args:
            maze: 2D list representing the maze (1=wall, 0=path)
        """
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)
        self.start = (1, 0)
        self.end = (self.height-2, self.width-1)
        
        # Initialize the specific solvers
        self.dfs_solver = DFSMazeSolver(maze)
        self.bfs_solver = BFSMazeSolver(maze)
        self.astar_solver = AStarMazeSolver(maze)
        
        # Initialize the visualizer
        self.visualizer = MazeSolutionVisualizer(maze, self.start, self.end)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Delegate to the base solver implementation."""
        return self.dfs_solver.get_neighbors(pos)
    
    def manhattan_distance(self, pos: Tuple[int, int]) -> float:
        """Delegate to the A* solver implementation."""
        return self.astar_solver.manhattan_distance(pos)
    
    def euclidean_distance(self, pos: Tuple[int, int]) -> float:
        """Delegate to the A* solver implementation."""
        return self.astar_solver.euclidean_distance(pos)
    
    def diagonal_distance(self, pos: Tuple[int, int]) -> float:
        """Delegate to the A* solver implementation."""
        return self.astar_solver.diagonal_distance(pos)
    
    def dfs(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """Run the DFS algorithm."""
        return self.dfs_solver.solve()
    
    def bfs(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """Run the BFS algorithm."""
        return self.bfs_solver.solve()
    
    def astar(self, heuristic: str = 'manhattan') -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """Run the A* algorithm with the specified heuristic."""
        return self.astar_solver.solve(heuristic)
    
    def visualize_path(self, path: List[Tuple[int, int]]):
        """Visualize the path in the terminal."""
        self.visualizer.visualize_path_terminal(path)
    
    def visualize_search(self, path: List[Tuple[int, int]], visited: Set[Tuple[int, int]], 
                         color: str = 'blue', title: str = 'Maze Solution', block: bool = False):
        """Visualize the search results."""
        self.visualizer.visualize_search(path, visited, color, title, block)
    
    def visualize_all_searches(self, algorithm_results, colors, titles):
        """Visualize multiple search results."""
        self.visualizer.visualize_all_searches(algorithm_results, colors, titles)


# Example usage
if __name__ == "__main__":
    from maze_generator import Maze
    
    # Create a maze
    size = 150  # More reasonable size for visualization
    maze = Maze(size, size)
    maze = maze.generate_non_perfect(removal_percentage=0.3)
    
    # Create a solver
    solver = MazeSolver(maze)   
    
    # Test all algorithms with visualization
    algorithms = [
        ('DFS', solver.dfs, 'red'),
        ('BFS', solver.bfs, 'blue'),
        ('A* (Manhattan)', lambda: solver.astar('manhattan'), 'green'),
        ('A* (Euclidean)', lambda: solver.astar('euclidean'), 'purple'),
        ('A* (Diagonal)', lambda: solver.astar('diagonal'), 'orange')
    ]
    
    # Run all algorithms and collect results
    results = []
    for name, algorithm, color in algorithms:
        print(f"Running {name}...")
        result = algorithm()
        results.append(result)
        
        print(f"{name} Metrics:")
        print(f"Nodes explored: {result[1]['nodes_explored']}")
        print(f"Time taken: {result[1]['time_taken']:.4f} seconds")
        print(f"Path length: {result[1]['path_length']}")
    
    # Display all results in a single figure
    colors = [algo[2] for algo in algorithms]
    titles = [algo[0] for algo in algorithms]
    solver.visualize_all_searches(results, colors, titles)
    
    # Optionally, if you want individual visualizations too:
    if False:  # Set to True to see individual visualizations
        for i, ((name, _, color), result) in enumerate(zip(algorithms, results)):
            path, metrics, visited = result
            # Block only on the last figure
            is_last = i == len(algorithms) - 1
            solver.visualize_search(path, visited, color=color, title=f"{name} Solution", block=is_last)
    
    print("All algorithms completed!") 