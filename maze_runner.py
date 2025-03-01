#!/usr/bin/env python3
"""
Maze Runner - Main script for running and comparing maze solving algorithms.
This script provides a unified interface to run all maze solving algorithms
and compare their performance.
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set
from maze_generator import Maze
from maze_solvers import MazeSolver, DFSMazeSolver, BFSMazeSolver, AStarMazeSolver, MazeVisualizer
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
                      metrics_to_compare: List[str] = ['nodes_explored', 'time_taken', 'path_length'],
                      maze_type: str = 'perfect', maze_params: Dict = {}):
    """
    Compare multiple algorithms across different maze sizes.
    
    Args:
        maze_sizes: List of maze sizes to test
        algorithms: List of tuples (name, solver_func, kwargs, color)
        metrics_to_compare: List of metric names to compare
        maze_type: Type of maze to generate ('perfect' or 'non_perfect')
        maze_params: Parameters for maze generation (e.g., removal_percentage)
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
        elif maze_type == 'non_perfect':
            removal_percentage = maze_params.get('removal_percentage', 0.1)
            maze = maze_gen.generate_non_perfect(removal_percentage=removal_percentage)
            print(f"Generated a non-perfect maze with {removal_percentage*100:.1f}% walls removed")
        else:
            print(f"Unknown maze type: {maze_type}. Using perfect maze instead.")
            maze = maze_gen.generate()
        
        # Analyze the maze
        maze_analysis = maze_gen.analyze_maze()
        print(f"Maze analysis:")
        print(f"  - Shortest path length: {maze_analysis['shortest_path_length']}")
        print(f"  - Is perfect maze: {maze_analysis['is_perfect']}")
        print(f"  - Number of paths: {maze_analysis['number_of_paths']}")
        
        # Create a visualizer for this maze
        start_pos = (1, 0)
        end_pos = (size-2, size-1)
        visualizer = MazeVisualizer(maze, start_pos, end_pos)
        
        # Run each algorithm
        algorithm_results = []
        algorithm_names = []
        algorithm_colors = []
        
        for name, solver_func, kwargs, color in algorithms:
            # Skip MDP algorithms for large mazes
            if ('Value Iteration' in name or 'Policy Iteration' in name) and size > 51:
                print(f"Skipping {name} for size {size} (too large for MDP algorithms)")
                continue
                
            try:
                result = run_algorithm(name, solver_func, maze=maze, **kwargs)
                results[size][name] = result[1]  # Store metrics
                algorithm_results.append(result)
                algorithm_names.append(name)
                algorithm_colors.append(color)
            except Exception as e:
                print(f"Error running {name} on size {size}: {e}")
                results[size][name] = {'error': str(e)}
        
        # Visualize all results for this maze size
        if algorithm_results:
            print("About to visualize results...")
            visualizer.visualize_all_searches(algorithm_results, algorithm_colors, algorithm_names)
            print("Visualization complete")
    
    # Create comparison plots for each metric
    for metric in metrics_to_compare:
        plt.figure(figsize=(10, 6))
        
        for name, _, _, color in algorithms:
            sizes = []
            values = []
            
            for size in maze_sizes:
                if name in results[size] and metric in results[size][name] and not isinstance(results[size][name].get('error', None), str):
                    sizes.append(size)
                    values.append(results[size][name][metric])
            
            if sizes:
                plt.plot(sizes, values, 'o-', label=name, color=color)
        
        plt.xlabel('Maze Size')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric.replace("_", " ").title()} across Algorithms')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'comparison_{maze_type}_{metric}.png')
        plt.show()
    
    return results


def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description='Run and compare maze solving algorithms')
    
    # Maze size options
    parser.add_argument('--sizes', type=int, nargs='+', default=[25, 51, 101],
                        help='Maze sizes to test (odd numbers recommended)')
    
    # Algorithm selection
    parser.add_argument('--search', action='store_true', help='Run search algorithms (DFS, BFS, A*)')
    parser.add_argument('--mdp', action='store_true', help='Run MDP algorithms (Value Iteration, Policy Iteration)')
    parser.add_argument('--all', action='store_true', help='Run all algorithms')
    
    # Visualization options
    parser.add_argument('--visualize-only', action='store_true', 
                        help='Only visualize a single maze solution')
    parser.add_argument('--size', type=int, default=31, 
                        help='Maze size for visualization only')
    
    # Maze type options
    parser.add_argument('--maze-type', type=str, choices=['perfect', 'non_perfect'], 
                        default='perfect', help='Type of maze to generate')
    parser.add_argument('--removal-percentage', type=float, default=0.1,
                        help='Percentage of internal walls to remove (0.0 to 1.0) for non-perfect mazes')
    
    args = parser.parse_args()
    
    print("Starting maze_runner.py")
    
    # If no algorithm type is specified, default to all
    if not (args.search or args.mdp) and not args.all:
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
            ('DFS', lambda maze: DFSMazeSolver(maze).solve(), {}, 'red'),
            ('BFS', lambda maze: BFSMazeSolver(maze).solve(), {}, 'blue'),
            ('A* (Manhattan)', lambda maze: AStarMazeSolver(maze).solve(heuristic='manhattan'), {}, 'green'),
            ('A* (Euclidean)', lambda maze: AStarMazeSolver(maze).solve(heuristic='euclidean'), {}, 'purple'),
            ('A* (Diagonal)', lambda maze: AStarMazeSolver(maze).solve(heuristic='diagonal'), {}, 'orange')
        ])
    
    if args.mdp:
        print("Adding MDP algorithms")
        # MDP algorithms with optimized parameters
        algorithms.extend([
            ('Value Iteration', lambda maze: ValueIterationSolver(
                maze, 
                discount_factor=0.95, 
                reward_exit=10.0,
                reward_step=-0.01
            ).solve(epsilon=0.001, max_iterations=200), {}, 'brown'),
            ('Policy Iteration', lambda maze: PolicyIterationSolver(
                maze,
                discount_factor=0.95,
                reward_exit=10.0,
                reward_step=-0.01
            ).solve(max_iterations=50, policy_eval_iterations=10), {}, 'teal')
        ])
    
    # Maze generation parameters
    maze_params = {
        'removal_percentage': args.removal_percentage
    }
    
    print("Command parameters:")
    print(f"  - visualize-only: {args.visualize_only}")
    print(f"  - maze-type: {args.maze_type}")
    print(f"  - removal-percentage: {args.removal_percentage}")
    print(f"  - search: {args.search}")
    print(f"  - mdp: {args.mdp}")
    
    # If visualize-only flag is set, just visualize a single maze with all algorithms
    if args.visualize_only:
        size = args.size
        print(f"Visualizing maze of size {size}x{size} with all selected algorithms")
        print(f"Maze type: {args.maze_type}")
        
        # Generate maze
        print("Generating maze...")
        maze_gen = Maze(size, size)
        if args.maze_type == 'perfect':
            maze = maze_gen.generate()
        elif args.maze_type == 'non_perfect':
            maze = maze_gen.generate_non_perfect(removal_percentage=args.removal_percentage)
        print("Maze generation complete")
        
        # Analyze maze
        print("Analyzing maze...")
        analysis = maze_gen.analyze_maze()
        print(f"Maze analysis:")
        print(f"  - Shortest path length: {analysis['shortest_path_length']}")
        print(f"  - Is perfect maze: {analysis['is_perfect']}")
        print(f"  - Number of paths: {analysis['number_of_paths']}")
        
        # Find the optimal path
        print("Finding shortest path...")
        optimal_path = maze_gen._find_shortest_path()
        print(f"Shortest path found with length: {len(optimal_path)}")
        
        # Create a visualizer
        start_pos = (1, 0)
        end_pos = (size-2, size-1)
        visualizer = MazeVisualizer(maze, start_pos, end_pos)
        
        # Run each algorithm
        algorithm_results = []
        algorithm_names = []
        algorithm_colors = []
        
        for name, solver_func, kwargs, color in algorithms:
            # Skip MDP algorithms for large mazes
            if ('Value Iteration' in name or 'Policy Iteration' in name) and size > 51:
                print(f"Skipping {name} for size {size} (too large for MDP algorithms)")
                continue
                
            try:
                print(f"Running {name}...")
                result = solver_func(maze=maze, **kwargs)
                algorithm_results.append(result)
                algorithm_names.append(name)
                algorithm_colors.append(color)
                
                # Print metrics
                path, metrics, visited = result
                print(f"{name} Metrics:")
                print(f"Nodes explored: {metrics['nodes_explored']}")
                print(f"Time taken: {metrics['time_taken']:.4f} seconds")
                print(f"Path length: {metrics['path_length']}")
                
                # Check if found path is optimal
                if metrics['path_length'] > 0:
                    is_optimal = metrics['path_length'] == analysis['shortest_path_length']
                    print(f"Found path is optimal: {is_optimal}")
                
                if 'iterations' in metrics:
                    print(f"Iterations: {metrics['iterations']}")
                print()
            except Exception as e:
                print(f"Error running {name}: {e}")
        
        # Visualize all results
        if algorithm_results:
            print("About to visualize algorithm results...")
            visualizer.visualize_all_searches(algorithm_results, algorithm_colors, algorithm_names)
            print("Visualization of algorithm results complete")
            
            # Also show the maze with the optimal path highlighted
            print("Showing maze with optimal path...")
            maze_gen.show_maze(highlight_path=optimal_path)
            print("Maze visualization complete")
    else:
        # Run comparison across all specified maze sizes
        print("Running algorithm comparison across multiple maze sizes")
        compare_algorithms(
            args.sizes, 
            algorithms, 
            maze_type=args.maze_type,
            maze_params=maze_params
        )
    
    print("maze_runner.py execution complete")


if __name__ == "__main__":
    main() 