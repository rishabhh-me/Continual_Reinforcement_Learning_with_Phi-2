import heapq
import numpy as np
from minigrid.core.constants import DIR_TO_VEC

class BFSController:
    def __init__(self, env):
        self.env = env
        self.grid = env.unwrapped.grid
        self.width = self.grid.width
        self.height = self.grid.height

    def get_path_to_pos(self, start_pos, start_dir, target_pos):
        """
        Finds a sequence of actions to go from start_pos/dir to target_pos.
        Does not account for dynamic objects blocking, assumes static grid map check.
        """
        # State: (x, y, dir)
        start_state = (start_pos[0], start_pos[1], start_dir)
        
        # Queue: (cost, state, path)
        queue = [(0, start_state, [])]
        visited = set()
        visited.add(start_state)

        while queue:
            cost, (x, y, d), path = heapq.heappop(queue)

            if (x, y) == target_pos:
                return path

            # Actions: 0=left, 1=right, 2=forward
            # MiniGrid actions: left=0, right=1, forward=2 (in agent view)
            # Actually MiniGrid actions constants: left=0, right=1, forward=2.
            
            # Try Forward
            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                cell = self.grid.get(nx, ny)
                # Passable if None or (object and type not in blocking)
                # For navigation, we treat doors as passable if open, strictly blocking if closed?
                # Actually for "GoTo" we might want to stop ADJACENT to target.
                # But let's assume we want to occupy the tile if possible, or handle adjacency later.
                
                is_passable = (cell is None) or (cell.type in ['goal', 'key', 'ball', 'box']) or (cell.type == 'door' and cell.is_open)
                
                if is_passable:
                    new_state = (nx, ny, d)
                    if new_state not in visited:
                        visited.add(new_state)
                        heapq.heappush(queue, (cost + 1, new_state, path + [self.env.unwrapped.actions.forward]))

            # Try Left
            nd = (d - 1) % 4
            new_state = (x, y, nd)
            if new_state not in visited:
                visited.add(new_state)
                heapq.heappush(queue, (cost + 1, new_state, path + [self.env.unwrapped.actions.left]))

            # Try Right
            nd = (d + 1) % 4
            new_state = (x, y, nd)
            if new_state not in visited:
                visited.add(new_state)
                heapq.heappush(queue, (cost + 1, new_state, path + [self.env.unwrapped.actions.right]))

        return None # No path found

    def get_path_to_adjacent(self, start_pos, start_dir, target_pos):
        """
        Finds path to be adjacent to target_pos and facing it.
        Useful for interacting (Open, Pick).
        """
        # We want to reach a state (tx, ty, tdir) such that (tx+dx, ty+dy) == target_pos
        
        # Valid target states:
        target_states = []
        for d in range(4):
            dx, dy = DIR_TO_VEC[d]
            tx, ty = target_pos[0] - dx, target_pos[1] - dy
            if 0 <= tx < self.width and 0 <= ty < self.height:
                cell = self.grid.get(tx, ty)
                # Must be passable to stand there
                if (cell is None) or (cell.type in ['goal', 'key', 'ball', 'box']) or (cell.type == 'door' and cell.is_open):
                     target_states.append((tx, ty, d))

        if not target_states:
            return None

        # BFS to any of target_states
        start_state = (start_pos[0], start_pos[1], start_dir)
        queue = [(0, start_state, [])]
        visited = set()
        visited.add(start_state)

        while queue:
            cost, state, path = heapq.heappop(queue)

            if state in target_states:
                return path

            x, y, d = state
            
            # Forward
            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                cell = self.grid.get(nx, ny)
                is_passable = (cell is None) or (cell.type in ['goal', 'key', 'ball', 'box']) or (cell.type == 'door' and cell.is_open)
                if is_passable:
                    new_state = (nx, ny, d)
                    if new_state not in visited:
                        visited.add(new_state)
                        heapq.heappush(queue, (cost + 1, new_state, path + [self.env.unwrapped.actions.forward]))

            # Left
            nd = (d - 1) % 4
            new_state = (x, y, nd)
            if new_state not in visited:
                visited.add(new_state)
                heapq.heappush(queue, (cost + 1, new_state, path + [self.env.unwrapped.actions.left]))

            # Right
            nd = (d + 1) % 4
            new_state = (x, y, nd)
            if new_state not in visited:
                visited.add(new_state)
                heapq.heappush(queue, (cost + 1, new_state, path + [self.env.unwrapped.actions.right]))

        return None

def get_low_level_plan(env, subgoal_tuple):
    """
    Returns a list of low-level actions to achieve the subgoal.
    """
    action_type, color, obj_type = subgoal_tuple
    grid = env.unwrapped.grid
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    bfs = BFSController(env)

    target_pos = None
    
    # Find target object position
    for i in range(grid.width):
        for j in range(grid.height):
            cell = grid.get(i, j)
            if cell and cell.type == obj_type:
                # If color is specified and not 'any', check color
                if color != 'any' and cell.color != color:
                    continue
                # Found it
                target_pos = (i, j)
                break
        if target_pos: break
    
    if not target_pos:
        return []

    actions = []
    
    if action_type == "pick":
        # Go to position
        path = bfs.get_path_to_pos(agent_pos, agent_dir, target_pos)
        if path is not None:
            actions.extend(path)
            actions.append(env.unwrapped.actions.pickup)
    
    elif action_type == "open":
        # Go to adjacent (facing)
        path = bfs.get_path_to_adjacent(agent_pos, agent_dir, target_pos)
        if path is not None:
            actions.extend(path)
            actions.append(env.unwrapped.actions.toggle)

    elif action_type == "goto":
        # Go to position
        path = bfs.get_path_to_pos(agent_pos, agent_dir, target_pos)
        if path is not None:
            actions.extend(path)
            # No interaction for goto
            
    return actions
