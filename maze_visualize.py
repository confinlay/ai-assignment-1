#!/usr/bin/env python3
"""
Maze Visualizer - Script for visualizing maze solving algorithms.
This script provides command line arguments to visualize different
maze solving algorithms on a maze of specified size.

This script can visualize the following maze solving algorithms:

1. Search Algorithms:
   - Depth-First Search (DFS): Explores as far as possible along each branch before backtracking
   - Breadth-First Search (BFS): Explores all neighbors at the present depth before moving to nodes at the next depth
   - A* Search: Uses heuristics to find the shortest path more efficiently
     - Manhattan distance: |x1-x2| + |y1-y2|
     - Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)
     - Diagonal distance: max(|x1-x2|, |y1-y2|)

2. Markov Decision Process (MDP) Algorithms:
   - Value Iteration: Computes the optimal state values and derives a policy
   - Policy Iteration: Iteratively improves a policy by policy evaluation and improvement

Command-line arguments:
  --size: Maze size for visualization [default: 31]
  --search: Run only search algorithms (DFS, BFS, A*)
  --mdp: Run only MDP algorithms (Value Iteration, Policy Iteration)
  --include-heuristics: Include all A* heuristics (manhattan, euclidean, diagonal)
  --maze-type: Type of maze to generate (perfect, imperfect) [default: perfect]
  --removal-percentage: Percentage of walls to remove for non-perfect mazes [default: 0.1]
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set
from maze_generator import Maze
from maze_solvers import MazeSolver, MazeSolutionVisualizer
from mdp_solvers import ValueIterationSolver, PolicyIterationSolver


def run_algorithm(name: str, solver_func, **kwargs) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
    """
    Run a maze solving algorithm and print its metrics.
    
    Args:
        name: Name of the algorithm
        solver_func: Function to call to solve the maze
        **kwargs: Additional arguments to pass to the solver function
        
    Returns:
        The result of the solver function (path, metrics, visited)
    """
    print(f"Running {name}...")
    try:
        result = solver_func(**kwargs)
        path, metrics, visited = result
        
        print(f"{name} Metrics:")
        print(f"Nodes explored: {metrics.get('nodes_explored', 'N/A')}")
        print(f"Time taken: {metrics.get('time_taken', 0):.4f} seconds")
        print(f"Path length: {len(path) if path else 0}")
        if 'iterations' in metrics:
            print(f"Iterations: {metrics['iterations']}")
        print()
        
        return result
    except Exception as e:
        print(f"Error running {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_algorithms(maze_size: int, algorithms, maze_type: str = 'perfect', maze_params: Dict = {}):
    """
    Visualize multiple algorithms on a maze of specified size.
    
    Args:
        maze_size: Size of the maze to generate
        algorithms: List of tuples with (name, function, params, color) for each algorithm
        maze_type: Type of maze to generate ('perfect' or 'imperfect')
        maze_params: Parameters for maze generation (e.g., removal_percentage)
    """
    print(f"\n{'='*50}")
    print(f"Visualizing maze size: {maze_size}x{maze_size} with maze type: {maze_type}")
    print(f"{'='*50}\n")
    
    # Generate maze
    maze_gen = Maze(maze_size, maze_size)
    if maze_type == 'perfect':
        maze = maze_gen.generate()
        print("Generated a perfect maze (single solution path)")
    elif maze_type == 'imperfect':
        removal_percentage = maze_params.get('removal_percentage', 0.1)
        maze = maze_gen.generate_imperfect(removal_percentage=removal_percentage)
        print(f"Generated an imperfect maze with {removal_percentage*100:.1f}% walls removed")
    else:
        print(f"Unknown maze type: {maze_type}. Using perfect maze instead.")
        maze = maze_gen.generate()
    
    # Create a MazeSolutionVisualizer instance
    visualizer = MazeSolutionVisualizer(maze, maze_gen.entrance, maze_gen.exit)
    
    # Run algorithms using run_algorithm function
    results = []
    colors = []
    titles = []
    
    # Run selected algorithms
    for name, func, params, color in algorithms:
        result = run_algorithm(name, func, maze=maze)
        if result:
            path, metrics, visited = result
            results.append((path, metrics, visited))
            colors.append(color)
            titles.append(name)
    
    # Visualize all results in a single figure
    if results:
        visualizer.visualize_all_searches(results, colors, titles)
    else:
        print("No algorithms were successfully run to visualize.")


def main():
    """Main function to parse arguments and run the visualization."""
    parser = argparse.ArgumentParser(description='Visualize maze solving algorithms')
    
    # Maze size for visualization
    parser.add_argument('--size', type=int, default=31, 
                        help='Maze size for visualization (odd numbers recommended)')
    
    # Algorithm selection
    parser.add_argument('--search', action='store_true', help='Run only search algorithms (DFS, BFS, A*)')
    parser.add_argument('--mdp', action='store_true', help='Run only MDP algorithms (Value Iteration, Policy Iteration)')
    parser.add_argument('--include-heuristics', action='store_true', 
                        help='Include all A* heuristics (manhattan, euclidean, diagonal)')
    
    # Maze type options
    parser.add_argument('--maze-type', type=str, choices=['perfect', 'imperfect'], 
                        default='perfect', help='Type of maze to generate')
    parser.add_argument('--removal-percentage', type=float, default=0.1,
                        help='Percentage of internal walls to remove (0.0 to 1.0) for non-perfect mazes')
    
    args = parser.parse_args()
    
    print("Starting maze_visualize.py")
    
    # If no algorithm type is specified, run all
    run_search = args.search or not (args.search or args.mdp)
    run_mdp = args.mdp or not (args.search or args.mdp)
    
    # Define algorithms to run
    algorithms = []
    
    if run_search:
        print("Adding search algorithms")
        # Search algorithms
        algorithms.extend([
            ('DFS', lambda maze: MazeSolver(maze).dfs(), {}, 'red'),
            ('BFS', lambda maze: MazeSolver(maze).bfs(), {}, 'blue'),
        ])
        
        # Add A* with manhattan heuristic by default, or all heuristics if specified
        if args.include_heuristics:
            algorithms.extend([
                ('A* (Manhattan)', lambda maze: MazeSolver(maze).astar('manhattan'), {}, 'green'),
                ('A* (Euclidean)', lambda maze: MazeSolver(maze).astar('euclidean'), {}, 'purple'),
                ('A* (Diagonal)', lambda maze: MazeSolver(maze).astar('diagonal'), {}, 'orange')
            ])
        else:
            algorithms.append(('A* (Manhattan)', lambda maze: MazeSolver(maze).astar('manhattan'), {}, 'green'))
    
    if run_mdp:
        print("Adding MDP algorithms")
        # MDP algorithms with optimized parameters
        algorithms.extend([
            ('Value Iteration', lambda maze: ValueIterationSolver(maze).solve(), {}, 'brown'),
            ('Policy Iteration', lambda maze: PolicyIterationSolver(maze).solve(), {}, 'teal')
        ])
    
    # Maze generation parameters
    maze_params = {
        'removal_percentage': args.removal_percentage
    }
    
    print("Command parameters:")
    print(f"  - size: {args.size}")
    print(f"  - maze-type: {args.maze_type}")
    print(f"  - removal-percentage: {args.removal_percentage}")
    print(f"  - search: {args.search}")
    print(f"  - mdp: {args.mdp}")
    print(f"  - include-heuristics: {args.include_heuristics}")
    
    # Run visualization
    visualize_algorithms(
        args.size, 
        algorithms,  # Pass the full algorithms list instead of just names
        maze_type=args.maze_type,
        maze_params=maze_params
    )
    
    print("maze_visualize.py execution complete")


if __name__ == "__main__":
    main()