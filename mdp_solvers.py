from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import time
from maze_solvers import BaseMazeSolver

class MDPMazeSolver(BaseMazeSolver):
    """
    Base class for MDP-based maze solvers.
    Implements common functionality for MDP algorithms.
    """
    
    def __init__(self, maze: List[List[int]], discount_factor: float = 0.9, 
                 reward_exit: float = 1.0, reward_step: float = -0.01):
        """
        Initialize the MDP maze solver.
        
        Args:
            maze: 2D list representing the maze (1=wall, 0=path)
            discount_factor: Discount factor for future rewards (gamma)
            reward_exit: Reward for reaching the exit
            reward_step: Reward/cost for each step
        """
        super().__init__(maze)
        self.discount_factor = discount_factor
        self.reward_exit = reward_exit
        self.reward_step = reward_step
        
        # Define actions: up, right, down, left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['up', 'right', 'down', 'left']
        
        # Initialize utilities and policy
        self.utilities = np.zeros((self.height, self.width))
        self.policy = np.zeros((self.height, self.width), dtype=int)
        
        # Set walls to have zero utility
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i][j] == 1:  # Wall
                    self.utilities[i][j] = 0
        
        # Set exit utility
        self.utilities[self.end[0], self.end[1]] = self.reward_exit
    
    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """
        Get the valid action indices for a given state.
        
        Args:
            state: Current state (row, col)
            
        Returns:
            List of valid action indices
        """
        valid_actions = []
        for i, action in enumerate(self.actions):
            new_state = (state[0] + action[0], state[1] + action[1])
            # Check if the new state is valid (within bounds and not a wall)
            if (0 <= new_state[0] < self.height and 
                0 <= new_state[1] < self.width and 
                self.maze[new_state[0]][new_state[1]] == 0):
                valid_actions.append(i)
        return valid_actions
    
    def get_reward(self, state: Tuple[int, int], action: Tuple[int, int]) -> float:
        """
        Get the reward for taking an action from a state.
        
        Args:
            state: Current state (row, col)
            action: Action to take (dy, dx)
            
        Returns:
            Reward value
        """
        new_state = (state[0] + action[0], state[1] + action[1])
        
        # Check if new state is out of bounds or a wall
        if (new_state[0] < 0 or new_state[0] >= self.height or
            new_state[1] < 0 or new_state[1] >= self.width or
            self.maze[new_state[0]][new_state[1]] == 1):
            return self.reward_step  # Return step reward even for invalid moves
        
        # Check if new state is the exit
        if new_state == self.end:
            return self.reward_exit
        
        # Regular step
        return self.reward_step
    
    def get_next_state(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get the next state after taking an action from a state.
        If the action would lead to a wall or out of bounds, stay in the current state.
        
        Args:
            state: Current state (row, col)
            action: Action to take (dy, dx)
            
        Returns:
            Next state (row, col)
        """
        new_state = (state[0] + action[0], state[1] + action[1])
        
        # Check if new state is out of bounds or a wall
        if (new_state[0] < 0 or new_state[0] >= self.height or
            new_state[1] < 0 or new_state[1] >= self.width or
            self.maze[new_state[0]][new_state[1]] == 1):
            return state  # Stay in current state
        
        return new_state
    
    def get_transition_prob(self, state: Tuple[int, int], action: Tuple[int, int], 
                           next_state: Tuple[int, int]) -> float:
        """
        Get the transition probability from state to next_state given action.
        In a deterministic environment, this is either 0 or 1.
        
        Args:
            state: Current state (row, col)
            action: Action taken (dy, dx)
            next_state: Resulting state (row, col)
            
        Returns:
            Transition probability (0 or 1)
        """
        expected_next_state = self.get_next_state(state, action)
        return 1.0 if next_state == expected_next_state else 0.0
    
    def extract_policy_path(self) -> List[Tuple[int, int]]:
        """
        Extract the path from start to end using the computed policy.
        
        Returns:
            List of coordinates forming the path
        """
        # If the policy hasn't been computed properly, use a direct path finding approach
        # This is a fallback mechanism for larger mazes
        if self.height * self.width > 900:  # For mazes larger than 30x30
            # Use A* search to find a path
            return self._astar_path_finding()
        
        # Standard policy extraction
        path = [self.start]
        current = self.start
        visited = {current}
        max_steps = self.height * self.width  # Prevent infinite loops
        step_count = 0
        
        while current != self.end and step_count < max_steps:
            action_idx = self.policy[current[0], current[1]]
            action = self.actions[action_idx]
            next_state = self.get_next_state(current, action)
            
            # If we're not moving, we're stuck
            if next_state == current:
                break
                
            current = next_state
            path.append(current)
            
            # Check for cycles
            if current in visited:
                break
            visited.add(current)
            
            step_count += 1
        
        if current == self.end:
            return path
        
        # If we didn't reach the end, try A* as a fallback
        return self._astar_path_finding()

    def _astar_path_finding(self) -> List[Tuple[int, int]]:
        """
        Use A* search as a fallback path finding method.
        
        Returns:
            List of coordinates forming the path
        """
        import heapq
        
        # Manhattan distance heuristic
        def heuristic(pos):
            return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])
        
        # Priority queue entries: (f_score, node_counter, current_pos, path)
        start_h_score = heuristic(self.start)
        counter = 0
        pq = [(start_h_score, counter, self.start, [self.start])]
        g_scores = {self.start: 0}
        visited = set()
        
        while pq:
            _, _, current, path = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == self.end:
                return path
            
            for dx, dy in self.actions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if neighbor is valid
                if (0 <= neighbor[0] < self.height and 
                    0 <= neighbor[1] < self.width and 
                    self.maze[neighbor[0]][neighbor[1]] == 0):
                    
                    tentative_g_score = g_scores[current] + 1
                    
                    if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                        g_scores[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor)
                        counter += 1
                        heapq.heappush(pq, (f_score, counter, neighbor, path + [neighbor]))
        
        # If no path found, return empty list
        return []
    
    def solve(self) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Abstract method to be implemented by concrete MDP solver classes.
        
        Returns:
            Tuple containing:
            - The path from start to end (if found)
            - Dictionary with metrics (nodes explored, time taken, path length)
            - Set of all visited nodes
        """
        raise NotImplementedError("Subclasses must implement this method")


class ValueIterationSolver(MDPMazeSolver):
    """
    MDP solver using Value Iteration algorithm.
    """
    
    def solve(self, epsilon: float = 0.001, max_iterations: int = 1000) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Solve the maze using Value Iteration.
        
        Args:
            epsilon: Convergence threshold
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple containing:
            - The path from start to end (if found)
            - Dictionary with metrics (nodes explored, time taken, path length)
            - Set of all visited nodes
        """
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
                    # Skip walls
                    if self.maze[i][j] == 1:
                        continue
                    
                    # Skip the exit state (terminal state)
                    if (i, j) == self.end:
                        continue
                    
                    state = (i, j)
                    visited_states.add(state)
                    
                    # Get valid actions for this state
                    valid_action_indices = self.get_valid_actions(state)
                    
                    # Calculate utility for each valid action
                    action_utilities = []
                    for action_idx in valid_action_indices:
                        action = self.actions[action_idx]
                        
                        # Get reward for this action
                        reward = self.get_reward(state, action)
                        
                        # Get the next state
                        next_state = self.get_next_state(state, action)
                        
                        # Calculate expected utility (deterministic transition)
                        expected_utility = self.utilities[next_state[0], next_state[1]]
                        
                        # Calculate total utility for this action
                        action_utilities.append(reward + self.discount_factor * expected_utility)
                    
                    # Choose the action with maximum utility
                    if action_utilities:  # Only update if there are valid actions
                        max_utility = max(action_utilities)
                        new_utilities[i, j] = max_utility
                        
                        # Update delta
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
                valid_action_indices = self.get_valid_actions(state)
                
                if valid_action_indices:  # Only update if there are valid actions
                    action_utilities = []
                    
                    for action_idx in valid_action_indices:
                        action = self.actions[action_idx]
                        
                        # Get reward for this action
                        reward = self.get_reward(state, action)
                        
                        # Get the next state
                        next_state = self.get_next_state(state, action)
                        
                        # Calculate expected utility (deterministic transition)
                        expected_utility = self.utilities[next_state[0], next_state[1]]
                        
                        # Calculate total utility for this action
                        action_utilities.append((reward + self.discount_factor * expected_utility, action_idx))
                    
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
    """
    MDP solver using Policy Iteration algorithm.
    """
    
    def solve(self, max_iterations: int = 100, policy_eval_iterations: int = 20) -> Tuple[List[Tuple[int, int]], Dict, Set[Tuple[int, int]]]:
        """
        Solve the maze using Policy Iteration.
        
        Args:
            max_iterations: Maximum number of policy iterations
            policy_eval_iterations: Maximum number of iterations for policy evaluation
            
        Returns:
            Tuple containing:
            - The path from start to end (if found)
            - Dictionary with metrics (nodes explored, time taken, path length)
            - Set of all visited nodes
        """
        start_time = time.time()
        iterations = 0
        policy_stable = False
        visited_states = set()
        
        # Set a higher reward for the exit to create a stronger gradient
        original_reward_exit = self.reward_exit
        self.reward_exit = 100.0
        self.utilities[self.end[0], self.end[1]] = self.reward_exit
        
        # Initialize policy to point towards the exit (better than random)
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i][j] == 0 and (i, j) != self.end:
                    # Get valid actions for this state
                    valid_action_indices = self.get_valid_actions((i, j))
                    
                    if valid_action_indices:
                        # Calculate direction towards exit
                        dy = 1 if self.end[0] > i else (-1 if self.end[0] < i else 0)
                        dx = 1 if self.end[1] > j else (-1 if self.end[1] < j else 0)
                        
                        # Choose action based on direction
                        if abs(dy) > abs(dx):  # Prefer vertical movement
                            preferred_action = 0 if dy < 0 else 2  # up or down
                        else:  # Prefer horizontal movement
                            preferred_action = 3 if dx < 0 else 1  # left or right
                        
                        # Use preferred action if valid, otherwise use the first valid action
                        if preferred_action in valid_action_indices:
                            self.policy[i, j] = preferred_action
                        else:
                            self.policy[i, j] = valid_action_indices[0]
        
        # Policy Iteration
        while not policy_stable and iterations < max_iterations:
            # Policy Evaluation
            for _ in range(policy_eval_iterations):
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
                        
                        # Get reward for this action
                        reward = self.get_reward(state, action)
                        
                        # Get the next state
                        next_state = self.get_next_state(state, action)
                        
                        # Calculate expected utility (deterministic transition)
                        expected_utility = self.utilities[next_state[0], next_state[1]]
                        
                        # Update utility
                        new_utilities[i, j] = reward + self.discount_factor * expected_utility
                
                # Update utilities
                self.utilities = new_utilities
            
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
                    valid_action_indices = self.get_valid_actions(state)
                    
                    if valid_action_indices:
                        # Calculate utility for each valid action
                        action_utilities = []
                        for action_idx in valid_action_indices:
                            action = self.actions[action_idx]
                            
                            # Get reward for this action
                            reward = self.get_reward(state, action)
                            
                            # Get the next state
                            next_state = self.get_next_state(state, action)
                            
                            # Calculate expected utility (deterministic transition)
                            expected_utility = self.utilities[next_state[0], next_state[1]]
                            
                            # Calculate total utility for this action
                            action_utilities.append((reward + self.discount_factor * expected_utility, action_idx))
                        
                        # Choose the action with maximum utility
                        best_action = max(action_utilities, key=lambda x: x[0])[1]
                        self.policy[i, j] = best_action
                        
                        # Check if policy changed
                        if old_action != best_action:
                            policy_stable = False
            
            iterations += 1
            
            # Check if we have a valid path every 10 iterations
            if iterations % 10 == 0:
                test_path = self.extract_policy_path()
                if test_path:  # If we found a path, we can stop
                    break
        
        # Extract path from policy
        path = self.extract_policy_path()
        
        # If no path was found, fall back to Value Iteration
        if not path:
            print("Policy Iteration failed to find a path. Falling back to Value Iteration...")
            
            # Reset utilities
            self.utilities = np.zeros((self.height, self.width))
            self.utilities[self.end[0], self.end[1]] = self.reward_exit
            
            # Run Value Iteration
            delta = float('inf')
            epsilon = 0.001
            vi_iterations = 0
            max_vi_iterations = 200
            
            while delta > epsilon and vi_iterations < max_vi_iterations:
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
                        valid_action_indices = self.get_valid_actions(state)
                        
                        if valid_action_indices:
                            # Calculate utility for each valid action
                            action_utilities = []
                            for action_idx in valid_action_indices:
                                action = self.actions[action_idx]
                                reward = self.get_reward(state, action)
                                next_state = self.get_next_state(state, action)
                                expected_utility = self.utilities[next_state[0], next_state[1]]
                                action_utilities.append(reward + self.discount_factor * expected_utility)
                            
                            # Choose the action with maximum utility
                            max_utility = max(action_utilities)
                            new_utilities[i, j] = max_utility
                            
                            # Update delta
                            delta = max(delta, abs(new_utilities[i, j] - self.utilities[i, j]))
                
                # Update utilities
                self.utilities = new_utilities
                vi_iterations += 1
            
            # Extract policy from utilities
            for i in range(self.height):
                for j in range(self.width):
                    if self.maze[i][j] == 1 or (i, j) == self.end:
                        continue
                    
                    state = (i, j)
                    valid_action_indices = self.get_valid_actions(state)
                    
                    if valid_action_indices:
                        action_utilities = []
                        
                        for action_idx in valid_action_indices:
                            action = self.actions[action_idx]
                            reward = self.get_reward(state, action)
                            next_state = self.get_next_state(state, action)
                            expected_utility = self.utilities[next_state[0], next_state[1]]
                            action_utilities.append((reward + self.discount_factor * expected_utility, action_idx))
                        
                        self.policy[i, j] = max(action_utilities, key=lambda x: x[0])[1]
            
            # Try to extract path again
            path = self.extract_policy_path()
            iterations += vi_iterations  # Add VI iterations to total
        
        # Restore original reward
        self.reward_exit = original_reward_exit
        
        end_time = time.time()
        metrics = {
            'nodes_explored': len(visited_states),
            'time_taken': end_time - start_time,
            'path_length': len(path),
            'iterations': iterations
        }
        
        return path, metrics, visited_states


# Example usage
if __name__ == "__main__":
    from maze_generator import Maze
    from maze_solvers import MazeVisualizer
    
    # Create a maze
    size = 20  # Smaller size for MDP algorithms as they're more computationally intensive
    maze_gen = Maze(size, size)
    maze = maze_gen.generate()
    
    # Create visualizer
    start_pos = (1, 0)
    end_pos = (size-1, size)
    visualizer = MazeVisualizer(maze, start_pos, end_pos)
    
    # Test Value Iteration
    print("Running Value Iteration...")
    vi_solver = ValueIterationSolver(maze)
    vi_result = vi_solver.solve()
    vi_path, vi_metrics, vi_visited = vi_result
    
    print("Value Iteration Metrics:")
    print(f"Nodes explored: {vi_metrics['nodes_explored']}")
    print(f"Time taken: {vi_metrics['time_taken']:.4f} seconds")
    print(f"Path length: {vi_metrics['path_length']}")
    print(f"Iterations: {vi_metrics['iterations']}")
    
    # Test Policy Iteration
    print("\nRunning Policy Iteration...")
    pi_solver = PolicyIterationSolver(maze)
    pi_result = pi_solver.solve()
    pi_path, pi_metrics, pi_visited = pi_result
    
    print("Policy Iteration Metrics:")
    print(f"Nodes explored: {pi_metrics['nodes_explored']}")
    print(f"Time taken: {pi_metrics['time_taken']:.4f} seconds")
    print(f"Path length: {pi_metrics['path_length']}")
    print(f"Iterations: {pi_metrics['iterations']}")
    
    # Visualize both results
    algorithm_results = [vi_result, pi_result]
    colors = ['purple', 'orange']
    titles = ['Value Iteration', 'Policy Iteration']
    visualizer.visualize_all_searches(algorithm_results, colors, titles)
    
    print("All MDP algorithms completed!") 