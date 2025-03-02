from typing import List, Tuple, Dict, Set
import numpy as np
import time
from maze_solvers import BaseMazeSolver

class MDPMazeSolver(BaseMazeSolver):
    """Base class for MDP maze solvers."""
    
    def __init__(self, maze: List[List[int]], discount_factor: float = 0.99, 
                 reward_exit: float = 100.0, reward_step: float = 0.00):
        """Initialize the MDP maze solver."""
        super().__init__(maze)
        self.discount = discount_factor
        self.exit_reward = reward_exit
        self.step_reward = reward_step
        
        # Define actions: up, right, down, left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['up', 'right', 'down', 'left']
        
        # Initialize utilities and policy
        self.utilities = np.zeros((self.height, self.width))
        self.policy = np.zeros((self.height, self.width), dtype=int)
        
        # Set exit utility
        self.utilities[self.end[0], self.end[1]] = self.exit_reward
    
    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """Get the valid action indices for a given state."""
        valid_actions = []
        for i, action in enumerate(self.actions):
            new_state = (state[0] + action[0], state[1] + action[1])
            # Check if the new state is valid (within bounds and not a wall)
            if (0 <= new_state[0] < self.height and 
                0 <= new_state[1] < self.width and 
                self.maze[new_state[0]][new_state[1]] == 0):
                valid_actions.append(i)
        return valid_actions
    
    def get_next_state(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        """Get the next state after taking an action from a state."""
        new_state = (state[0] + action[0], state[1] + action[1])
        
        # Check if new state is out of bounds or a wall
        if (new_state[0] < 0 or new_state[0] >= self.height or
            new_state[1] < 0 or new_state[1] >= self.width or
            self.maze[new_state[0]][new_state[1]] == 1):
            return state  # Stay in current state
        
        return new_state
    
    def get_reward(self, state: Tuple[int, int], action: Tuple[int, int]) -> float:
        """Get the reward for taking an action from a state."""
        new_state = self.get_next_state(state, action)
        
        # Check if new state is the exit
        if new_state == self.end:
            return self.exit_reward
        
        # Regular step
        return self.step_reward
    
    def extract_policy_path(self) -> List[Tuple[int, int]]:
        """Extract the path from start to end using the computed policy."""
        path = [self.start]
        current = self.start
        visited = {current}
        max_steps = self.height * self.width  # Prevent infinite loops
        step_count = 0
        
        while current != self.end and step_count < max_steps:
            action_idx = self.policy[current[0], current[1]]
            action = self.actions[action_idx]
            next_state = self.get_next_state(current, action)
            
            # If we're stuck or in a cycle, stop
            if next_state == current or next_state in visited:
                break
                
            current = next_state
            path.append(current)
            visited.add(current)
            step_count += 1
        
        # Return the path if we reached the end, otherwise empty list
        return path if current == self.end else []
    
    def solve(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """Abstract method to be implemented by concrete MDP solver classes."""
        raise NotImplementedError("Subclasses must implement this method")


class ValueIterationSolver(MDPMazeSolver):
    """MDP solver using Value Iteration algorithm."""
    
    def solve(self, epsilon: float = 0.001, max_iterations: int = 2000) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """Solve the maze using Value Iteration."""
        start_time = time.time()
        iterations = 0
        delta = float('inf')
        visited_states = set()
        
        # Value Iteration
        while delta > epsilon and iterations < max_iterations:
            delta = 0
            new_utilities = np.copy(self.utilities)
            
            for i in range(self.height):
                for j in range(self.width):
                    # Skip walls and exit
                    if self.maze[i][j] == 1 or (i, j) == self.end:
                        continue
                    
                    state = (i, j)
                    visited_states.add(state)
                    
                    # Get valid actions for this state
                    valid_actions = self.get_valid_actions(state)
                    
                    # Calculate utility for each valid action
                    action_utilities = []
                    for action_idx in valid_actions:
                        action = self.actions[action_idx]
                        reward = self.get_reward(state, action)
                        next_state = self.get_next_state(state, action)
                        next_utility = self.utilities[next_state[0], next_state[1]]
                        action_utilities.append(reward + self.discount * next_utility)
                    
                    # Choose the action with maximum utility
                    if action_utilities:
                        max_utility = max(action_utilities)
                        new_utilities[i, j] = max_utility
                        
                        # Update delta for convergence check
                        delta = max(delta, abs(new_utilities[i, j] - self.utilities[i, j]))
            
            # Update utilities
            self.utilities = new_utilities
            iterations += 1
        
        # Extract policy from utilities
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i][j] == 1 or (i, j) == self.end:
                    continue
                
                state = (i, j)
                valid_actions = self.get_valid_actions(state)
                
                if valid_actions:
                    action_utilities = []
                    
                    for action_idx in valid_actions:
                        action = self.actions[action_idx]
                        reward = self.get_reward(state, action)
                        next_state = self.get_next_state(state, action)
                        next_utility = self.utilities[next_state[0], next_state[1]]
                        action_utilities.append((reward + self.discount * next_utility, action_idx))
                    
                    # Choose the action with maximum utility
                    self.policy[i, j] = max(action_utilities, key=lambda x: x[0])[1]
        
        # Extract path from policy
        path = self.extract_policy_path()
        
        end_time = time.time()
        metrics = {
            'nodes_explored': len(visited_states),
            'time_taken': end_time - start_time,
            'path_length': len(path),
            'iterations': iterations
        }
        
        return path, metrics, visited_states


class PolicyIterationSolver(MDPMazeSolver):
    """MDP solver using Policy Iteration algorithm."""
    
    def solve(self, max_iterations: int = 1000, theta: float = 0.001, max_eval_iterations: int = 1000) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """Solve the maze using Policy Iteration.
        
        Args:
            max_iterations: Maximum number of policy iteration cycles
            theta: Convergence threshold for policy evaluation
            max_eval_iterations: Maximum iterations for each policy evaluation phase
        """
        start_time = time.time()
        iterations = 0
        policy_stable = False
        visited_states = set()
        
        # Initialize policy randomly
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i][j] == 0 and (i, j) != self.end:
                    valid_actions = self.get_valid_actions((i, j))
                    if valid_actions:
                        self.policy[i, j] = np.random.choice(valid_actions)
        
        # Policy Iteration
        while not policy_stable and iterations < max_iterations:
            # Policy Evaluation (run until convergence)
            eval_iterations = 0
            delta = float('inf')
            
            while delta > theta and eval_iterations < max_eval_iterations:
                delta = 0
                new_utilities = np.copy(self.utilities)
                
                for i in range(self.height):
                    for j in range(self.width):
                        # Skip walls and exit
                        if self.maze[i][j] == 1 or (i, j) == self.end:
                            continue
                        
                        state = (i, j)
                        visited_states.add(state)
                        
                        # Get current policy action
                        action_idx = self.policy[i, j]
                        action = self.actions[action_idx]
                        
                        # Get reward and next state
                        reward = self.get_reward(state, action)
                        next_state = self.get_next_state(state, action)
                        
                        # Update utility
                        next_utility = self.utilities[next_state[0], next_state[1]]
                        new_value = reward + self.discount * next_utility
                        new_utilities[i, j] = new_value
                        
                        # Track maximum change
                        delta = max(delta, abs(new_value - self.utilities[i, j]))
                
                # Update utilities
                self.utilities = new_utilities
                eval_iterations += 1
            
            # Policy Improvement
            policy_stable = True
            
            for i in range(self.height):
                for j in range(self.width):
                    # Skip walls and exit
                    if self.maze[i][j] == 1 or (i, j) == self.end:
                        continue
                    
                    state = (i, j)
                    old_action = self.policy[i, j]
                    
                    # Get valid actions for this state
                    valid_actions = self.get_valid_actions(state)
                    
                    if valid_actions:
                        # Calculate utility for each valid action
                        action_utilities = []
                        for action_idx in valid_actions:
                            action = self.actions[action_idx]
                            reward = self.get_reward(state, action)
                            next_state = self.get_next_state(state, action)
                            next_utility = self.utilities[next_state[0], next_state[1]]
                            action_utilities.append((reward + self.discount * next_utility, action_idx))
                        
                        # Choose the action with maximum utility
                        best_action = max(action_utilities, key=lambda x: x[0])[1]
                        self.policy[i, j] = best_action
                        
                        # Check if policy changed
                        if old_action != best_action:
                            policy_stable = False
            
            iterations += 1
        
        # Extract path from policy
        path = self.extract_policy_path()
        
        end_time = time.time()
        metrics = {
            'nodes_explored': len(visited_states),
            'time_taken': end_time - start_time,
            'path_length': len(path),
            'iterations': iterations,
            'policy_eval_iterations': eval_iterations
        }
        
        return path, metrics, visited_states


# Example usage
if __name__ == "__main__":
    from maze_generator import Maze
    from maze_solvers import MazeSolutionVisualizer
    
    # Create a maze
    size = 150  # Small size for quick testing
    print(f"Creating a {size}x{size} maze...")
    maze_gen = Maze(size, size)
    maze = maze_gen.generate_imperfect(removal_percentage=0.1)
    
    # Create visualizer
    visualizer = MazeSolutionVisualizer(maze, maze_gen.entrance, maze_gen.exit)
    
    # Test Value Iteration
    print("\nRunning Value Iteration...")
    vi_solver = ValueIterationSolver(maze)
    vi_path, vi_metrics, vi_visited = vi_solver.solve()
    
    print("Value Iteration Results:")
    print(f"- Nodes explored: {vi_metrics['nodes_explored']}")
    print(f"- Time taken: {vi_metrics['time_taken']:.4f} seconds")
    print(f"- Path length: {vi_metrics['path_length']}")
    print(f"- Iterations: {vi_metrics['iterations']}")
    
    # Test Policy Iteration
    print("\nRunning Policy Iteration...")
    pi_solver = PolicyIterationSolver(maze)
    pi_path, pi_metrics, pi_visited = pi_solver.solve()
    
    print("Policy Iteration Results:")
    print(f"- Nodes explored: {pi_metrics['nodes_explored']}")
    print(f"- Time taken: {pi_metrics['time_taken']:.4f} seconds")
    print(f"- Path length: {pi_metrics['path_length']}")
    print(f"- Iterations: {pi_metrics['iterations']}")
    
    # Visualize both results
    algorithm_results = [(vi_path, vi_metrics, vi_visited), (pi_path, pi_metrics, pi_visited)]
    colors = ['purple', 'orange']
    titles = ['Value Iteration', 'Policy Iteration']
    visualizer.visualize_all_searches(algorithm_results, colors, titles)
    
    print("\nAll done!") 