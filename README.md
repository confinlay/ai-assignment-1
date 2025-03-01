# Maze Solving Algorithms

This project implements and compares various maze solving algorithms, including both search-based algorithms and Markov Decision Process (MDP) algorithms.

## Project Structure

The project is organized into the following files:

- `maze_generator.py`: Implements maze generation using the Recursive Backtracking algorithm
- `maze_solvers.py`: Contains search-based maze solving algorithms (DFS, BFS, A*)
- `mdp_solvers.py`: Contains MDP-based maze solving algorithms (Value Iteration, Policy Iteration)
- `maze_runner.py`: Main script for running and comparing all algorithms

## Algorithms Implemented

### Search Algorithms
1. **Depth-First Search (DFS)**: Explores as far as possible along each branch before backtracking
2. **Breadth-First Search (BFS)**: Explores all neighbors at the present depth before moving to nodes at the next depth
3. **A* Search**: Uses heuristics to guide the search towards the goal
   - Manhattan distance heuristic
   - Euclidean distance heuristic
   - Diagonal (Chebyshev) distance heuristic

### MDP Algorithms
1. **Value Iteration**: Iteratively computes the utility of each state until convergence
2. **Policy Iteration**: Alternates between policy evaluation and policy improvement steps

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install numpy matplotlib
   ```

## Usage

### Running All Algorithms

To run all algorithms on default maze sizes (25x25, 51x51, 101x101):

```bash
python maze_runner.py
```

### Specific Algorithm Types

To run only search algorithms:

```bash
python maze_runner.py --search
```

To run only MDP algorithms:

```bash
python maze_runner.py --mdp
```

### Custom Maze Sizes

To specify custom maze sizes:

```bash
python maze_runner.py --sizes 21 41 61
```

### Visualization Only

To only visualize the algorithms on a single maze without running the full comparison:

```bash
python maze_runner.py --visualize-only --size 31
```

## Metrics

The algorithms are compared using the following metrics:

1. **Nodes Explored**: Number of nodes/states visited during the search
2. **Time Taken**: Execution time in seconds
3. **Path Length**: Length of the solution path (if found)
4. **Iterations** (MDP only): Number of iterations until convergence

## Visualization

The project includes visualization tools to:

1. Display the maze with solution paths
2. Compare multiple algorithms on the same maze
3. Generate comparison plots for different metrics across maze sizes

## Example

```python
from maze_generator import Maze
from maze_solvers import MazeSolver

# Create a maze
size = 51
maze = Maze(size, size).generate()

# Create a solver
solver = MazeSolver(maze)

# Run DFS
path, metrics, visited = solver.dfs()
print(f"DFS Metrics: {metrics}")

# Visualize the solution
solver.visualize_path(path)
```

## Performance Considerations

- MDP algorithms are more computationally intensive and may be slow on larger mazes
- A* with an appropriate heuristic typically finds the shortest path
- BFS always finds the shortest path in unweighted graphs
- DFS may find a solution quickly but it's not guaranteed to be the shortest

## License

This project is open source and available under the MIT License. 