from typing import List, Tuple, Dict, Set, Callable, Optional
from collections import deque
import heapq
import time
import matplotlib.pyplot as plt
import numpy as np

class MazeSolver:
    """
    A class containing different algorithms for solving mazes.
    The maze is represented as a 2D list where:
    - 1 represents walls
    - 0 represents paths
    """
    
    def __init__(self, maze: List[List[int]]):
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)
        # Start position is always (1,0) in our maze
        self.start = (1, 0)
        # End position is always (height-2, width-1) in our maze
        self.end = (self.height-2, self.width-1)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns valid neighboring positions (up, right, down, left)."""
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

    # Heuristic functions
    def manhattan_distance(self, pos: Tuple[int, int]) -> float:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])
    
    def euclidean_distance(self, pos: Tuple[int, int]) -> float:
        """Calculate Euclidean distance heuristic (straight-line distance)."""
        return ((pos[0] - self.end[0]) ** 2 + (pos[1] - self.end[1]) ** 2) ** 0.5
    
    def diagonal_distance(self, pos: Tuple[int, int]) -> float:
        """
        Calculate Chebyshev distance heuristic.
        This is the maximum of the horizontal and vertical distances.
        """
        return max(abs(pos[0] - self.end[0]), abs(pos[1] - self.end[1]))

    def dfs(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Depth-First Search implementation.
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

    def bfs(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Breadth-First Search implementation.
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

    def astar(self, heuristic: str = 'manhattan') -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        A* Search implementation with configurable heuristic.
        
        Args:
            heuristic (str): The heuristic to use - 'manhattan', 'euclidean', or 'diagonal'
            
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

    def visualize_path(self, path: List[Tuple[int, int]]):
        """
        Visualize the maze with the solved path in the terminal.
        Path cells are marked with '◈◈'
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
            
        The path will be shown in the specified color, while visited nodes
        that are not part of the path will be shown in a lighter shade.
        """
        try:
            # Create a visualization array (0=path, 1=wall, 2=visited-not-in-path)
            viz_array = np.array(self.maze, dtype=np.int8)
            
            # Mark visited cells
            for y, x in visited:
                if viz_array[y, x] == 0:  # Only mark if it's a path
                    viz_array[y, x] = 2
            
            # Mark path cells
            for y, x in path:
                viz_array[y, x] = 3
                
            # Create a custom colormap
            from matplotlib.colors import ListedColormap
            
            # Create color maps based on the selected color
            import matplotlib.colors as mcolors
            base_color = mcolors.to_rgb(color)
            
            # Create lighter version for visited cells
            light_color = tuple([min(1.0, c + 0.5) for c in base_color])
            
            # Define the colormap: white, black, light_color, base_color
            cmap = ListedColormap(['white', 'black', light_color, base_color])
            
            # Create the figure
            plt.figure(figsize=(10, 10))
            
            # Plot the maze
            plt.imshow(viz_array, cmap=cmap)
            
            # Mark start and end
            plt.plot(self.start[1], self.start[0], 'go', markersize=8)  # Start in green
            plt.plot(self.end[1], self.end[0], 'ro', markersize=8)      # End in red
            
            # Remove axes and ticks
            plt.axis('off')
            
            # Add title and legend
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
            plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
            
            plt.tight_layout()
            plt.show(block=True)  # Always block to ensure the window is closed before continuing
            
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
            import matplotlib.colors as mcolors
            from matplotlib.colors import ListedColormap
            from matplotlib.patches import Patch
            
            # Setup the figure
            num_plots = len(algorithm_results)
            rows = (num_plots + 2) // 3  # Calculate number of rows needed (ceiling division)
            cols = min(3, num_plots)     # Max 3 columns
            
            fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
            if num_plots == 1:
                axs = [axs]  # Make it iterable for single plot
            else:
                axs = axs.flatten()  # Convert 2D array to 1D for easy iteration
            
            # Create plots for each algorithm
            for i, ((path, _, visited), color, title) in enumerate(zip(algorithm_results, colors, titles)):
                ax = axs[i]
                
                # Create a visualization array (0=path, 1=wall, 2=visited-not-in-path, 3=solution path)
                viz_array = np.array(self.maze, dtype=np.int8)
                
                # Mark visited cells
                for y, x in visited:
                    if viz_array[y, x] == 0:  # Only mark if it's a path
                        viz_array[y, x] = 2
                
                # Mark path cells
                for y, x in path:
                    viz_array[y, x] = 3
                    
                # Create color maps based on the selected color
                base_color = mcolors.to_rgb(color)
                light_color = tuple([min(1.0, c + 0.5) for c in base_color])
                
                # Define the colormap: white, black, light_color, base_color
                cmap = ListedColormap(['white', 'black', light_color, base_color])
                
                # Plot the maze
                ax.imshow(viz_array, cmap=cmap)
                
                # Mark start and end
                ax.plot(self.start[1], self.start[0], 'go', markersize=8)
                ax.plot(self.end[1], self.end[0], 'ro', markersize=8)
                
                # Remove axes and ticks
                ax.axis('off')
                
                # Add title with metrics
                ax.set_title(f"{title}\nPath Length: {len(path)}\nNodes Explored: {len(visited)}")
            
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

# Example usage
if __name__ == "__main__":
    from maze_generator import Maze
    
    # Create a maze
    size = 250  # Smaller size for quick visualization
    maze = Maze(size, size)
    maze = maze.generate()
    
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