#!/usr/bin/env python3
"""
Maze Runner - Main script for running and comparing maze solving algorithms.
This script provides many command line arguments to allow for easy comaprison of
the different maze solving algorithms under different conditions.

This script compares different maze solving algorithms:

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
  --sizes: List of maze sizes to test [default: 50, 100, 150, 200, 250]
  --search: Run search algorithms (DFS, BFS, A*)
  --mdp: Run MDP algorithms (Value Iteration, Policy Iteration)
  --all: Run all algorithms
  --astar-only: Run only A* algorithms with all heuristics
  --astar-heuristic: Specify a single A* heuristic (manhattan, euclidean, diagonal)
  --size: Maze size for visualization [default: 31]
  --maze-type: Type of maze to generate (perfect, imperfect) [default: perfect]
  --removal-percentage: Percentage of walls to remove for non-perfect mazes [default: 0.1]
  --csv: Export results to a CSV file

"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from typing import List, Dict, Tuple, Set
from maze_generator import Maze
from maze_solvers import MazeSolver
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
    start_time = time.time()
    result = solver_func(**kwargs)
    path, metrics, visited = result
    
    print(f"{name} Metrics:")
    print(f"Nodes explored: {metrics['nodes_explored']}")
    print(f"Time taken: {metrics['time_taken']:.4f} seconds")
    print(f"Path length: {metrics['path_length']}")
    if 'iterations' in metrics:
        print(f"Iterations: {metrics['iterations']}")
    print()
    
    return result


def compare_algorithms(maze_sizes: List[int], algorithms: List[Tuple[str, callable, Dict, str]], 
                      metrics_to_compare: List[str] = ['nodes_explored', 'time_taken'],
                      maze_type: str = 'perfect', maze_params: Dict = {}, export_csv: bool = False):
    """
    Compare multiple algorithms across different maze sizes.
    
    Args:
        maze_sizes: List of maze sizes to test
        algorithms: List of tuples (name, solver_func, kwargs, color)
        metrics_to_compare: List of metric names to compare
        maze_type: Type of maze to generate ('perfect' or 'imperfect')
        maze_params: Parameters for maze generation (e.g., removal_percentage)
        export_csv: Whether to export results to a CSV file
    """
    # Initialize results dictionary
    results = {size: {} for size in maze_sizes}
    
    # Run algorithms for each maze size
    for size in maze_sizes:
        print(f"\n{'='*50}")
        print(f"Testing maze size: {size}x{size} with maze type: {maze_type}")
        print(f"{'='*50}\n")
        
        # Generate maze
        maze_gen = Maze(size, size)
        if maze_type == 'perfect':
            maze = maze_gen.generate()
            print("Generated a perfect maze (single solution path)")
        elif maze_type == 'imperfect':
            removal_percentage = maze_params.get('removal_percentage', 0.1)
            maze = maze_gen.generate_imperfect(removal_percentage=removal_percentage)
            print(f"Generated a non-perfect maze with {removal_percentage*100:.1f}% walls removed")
        else:
            print(f"Unknown maze type: {maze_type}. Using perfect maze instead.")
            maze = maze_gen.generate()
        
        # Run each algorithm
        for name, solver_func, kwargs, color in algorithms:
            try:
                result = run_algorithm(name, solver_func, maze=maze, **kwargs)
                results[size][name] = result[1]  # Store metrics
            except Exception as e:
                print(f"Error running {name} on size {size}: {e}")
                results[size][name] = {'error': str(e)}
    
    # Create comparison plots for time_taken and nodes_explored
    plt.figure(figsize=(10, 6))
    for name, _, _, color in algorithms:
        sizes = []
        values = []
        
        for size in maze_sizes:
            if name in results[size] and 'time_taken' in results[size][name] and not isinstance(results[size][name].get('error', None), str):
                sizes.append(size)
                values.append(results[size][name]['time_taken'])
        
        if sizes:
            plt.plot(sizes, values, 'o-', label=name, color=color)
    
    plt.xlabel('Maze Size')
    plt.ylabel('Time Taken (s)')
    plt.title('Comparison of Time Taken across Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'comparison_{maze_type}_time_taken.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for name, _, _, color in algorithms:
        sizes = []
        values = []
        
        for size in maze_sizes:
            if name in results[size] and 'nodes_explored' in results[size][name] and not isinstance(results[size][name].get('error', None), str):
                sizes.append(size)
                values.append(results[size][name]['nodes_explored'])
        
        if sizes:
            plt.plot(sizes, values, 'o-', label=name, color=color)
    
    plt.xlabel('Maze Size')
    plt.ylabel('Nodes Explored')
    plt.title('Comparison of Nodes Explored across Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'comparison_{maze_type}_nodes_explored.png')
    plt.show()
    
    # Export results to CSV if requested
    if export_csv:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_filename = f'maze_results_{maze_type}_{timestamp}.csv'
        
        with open(csv_filename, 'w', newline='') as csvfile:
            # Determine all possible metrics across all results
            all_metrics = set()
            for size in results:
                for alg in results[size]:
                    if isinstance(results[size][alg], dict):
                        all_metrics.update(results[size][alg].keys())
            
            # Create fieldnames for CSV
            fieldnames = ['maze_size', 'algorithm'] + sorted(list(all_metrics))
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data rows
            for size in sorted(results.keys()):
                for alg in sorted(results[size].keys()):
                    row = {'maze_size': size, 'algorithm': alg}
                    if isinstance(results[size][alg], dict):
                        row.update(results[size][alg])
                    writer.writerow(row)
        
        print(f"Results exported to {csv_filename}")
    
    return results


def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description='Run and compare maze solving algorithms')
    
    # Maze size options
    parser.add_argument('--sizes', type=int, nargs='+', default=[50, 100, 150, 200, 250],
                        help='Maze sizes to test (odd numbers recommended)')
    
    # Algorithm selection
    parser.add_argument('--search', action='store_true', help='Run search algorithms (DFS, BFS, A*)')
    parser.add_argument('--mdp', action='store_true', help='Run MDP algorithms (Value Iteration, Policy Iteration)')
    parser.add_argument('--all', action='store_true', help='Run all algorithms')
    parser.add_argument('--astar-only', action='store_true', help='Run only A* algorithms with all heuristics')
    parser.add_argument('--astar-heuristic', type=str, choices=['manhattan', 'euclidean', 'diagonal'], 
                        help='Specify a single A* heuristic to use')
    
    # Maze size for single run
    parser.add_argument('--size', type=int, default=31, 
                        help='Maze size for single run')
    
    # Maze type options
    parser.add_argument('--maze-type', type=str, choices=['perfect', 'imperfect'], 
                        default='perfect', help='Type of maze to generate')
    parser.add_argument('--removal-percentage', type=float, default=0.1,
                        help='Percentage of internal walls to remove (0.0 to 1.0) for non-perfect mazes')
    
    # Export options
    parser.add_argument('--csv', action='store_true', help='Export results to a CSV file')
    
    args = parser.parse_args()
    
    print("Starting maze_runner.py")
    
    # If astar-only is specified, only run A* algorithms
    if args.astar_only:
        args.search = False
        args.mdp = False
        args.all = False
    # If no algorithm type is specified and not astar-only, default to all
    elif not (args.search or args.mdp) and not args.all:
        args.all = True
    
    # If --all is specified, run all algorithm types
    if args.all:
        args.search = True
        args.mdp = True
    
    # Define algorithms to run
    algorithms = []
    
    if args.search:
        print("Adding search algorithms")
        # Search algorithms
        algorithms.extend([
            ('DFS', lambda maze: MazeSolver(maze).dfs(), {}, 'red'),
            ('BFS', lambda maze: MazeSolver(maze).bfs(), {}, 'blue'),
        ])
        
        # Add A* with specified heuristic or all heuristics
        if args.astar_heuristic:
            algorithms.append((f'A* ({args.astar_heuristic.capitalize()})', 
                              lambda maze: MazeSolver(maze).astar(args.astar_heuristic), 
                              {}, 'green'))
        else:
            algorithms.extend([
                ('A* (Manhattan)', lambda maze: MazeSolver(maze).astar('manhattan'), {}, 'green'),
                ('A* (Euclidean)', lambda maze: MazeSolver(maze).astar('euclidean'), {}, 'purple'),
                ('A* (Diagonal)', lambda maze: MazeSolver(maze).astar('diagonal'), {}, 'orange')
            ])
    
    if args.mdp:
        print("Adding MDP algorithms")
        # MDP algorithms with optimized parameters
        algorithms.extend([
            ('Value Iteration', lambda maze: ValueIterationSolver(maze).solve(), {}, 'brown'),
            ('Policy Iteration', lambda maze: PolicyIterationSolver(maze).solve(), {}, 'teal')
        ])
    
    # If astar-only is specified, only run A* algorithms
    if args.astar_only:
        print("Adding only A* algorithms")
        if args.astar_heuristic:
            algorithms = [(f'A* ({args.astar_heuristic.capitalize()})', 
                          lambda maze: MazeSolver(maze).astar(args.astar_heuristic), 
                          {}, 'green')]
        else:
            algorithms = [
                ('A* (Manhattan)', lambda maze: MazeSolver(maze).astar('manhattan'), {}, 'green'),
                ('A* (Euclidean)', lambda maze: MazeSolver(maze).astar('euclidean'), {}, 'purple'),
                ('A* (Diagonal)', lambda maze: MazeSolver(maze).astar('diagonal'), {}, 'orange')
            ]
    
    # Maze generation parameters
    maze_params = {
        'removal_percentage': args.removal_percentage
    }
    
    print("Command parameters:")
    print(f"  - maze-type: {args.maze_type}")
    print(f"  - removal-percentage: {args.removal_percentage}")
    print(f"  - search: {args.search}")
    print(f"  - mdp: {args.mdp}")
    print(f"  - astar-only: {args.astar_only}")
    print(f"  - astar-heuristic: {args.astar_heuristic}")
    print(f"  - csv: {args.csv}")
    
    # Run comparison across all specified maze sizes
    print("Running algorithm comparison across multiple maze sizes")
    compare_algorithms(
        args.sizes, 
        algorithms, 
        maze_type=args.maze_type,
        maze_params=maze_params,
        export_csv=args.csv
    )
    
    print("maze_runner.py execution complete")


if __name__ == "__main__":
    main() 