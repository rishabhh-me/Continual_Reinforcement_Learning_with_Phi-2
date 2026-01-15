class Oracle:
    def __init__(self, env):
        self.env = env
    
    def get_next_subgoal(self):
        """
        Analyzes the current environment state and returns the next logical subgoal.
        Returns:
            subgoal_text (str): "Pick up the yellow key"
            subgoal_tuple (tuple): ("pick", "yellow", "key")
        """
        grid = self.env.unwrapped.grid
        agent_pos = self.env.unwrapped.agent_pos
        carrying = self.env.unwrapped.carrying
        
        # Find key, door, goal
        key_obj = None
        key_pos = None
        door_obj = None
        door_pos = None
        goal_pos = None
        
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell:
                    if cell.type == 'key':
                        key_obj = cell
                        key_pos = (i, j)
                    elif cell.type == 'door':
                        door_obj = cell
                        door_pos = (i, j)
                    elif cell.type == 'goal':
                        goal_pos = (i, j)

        # Logic for DoorKey / KeyCorridor
        
        # 1. Do we have the key?
        has_key = (carrying and carrying.type == 'key')
        
        # If we don't have the key:
        if not has_key:
            # If there is a closed door, we likely need the key.
            # (Assuming key matches door, which is true in simple DoorKey)
            if door_obj and not door_obj.is_open:
                # If key is visible/known
                if key_obj:
                    return f"Pick up the {key_obj.color} key", ("pick", key_obj.color, "key")
                else:
                    # Key not visible? Explore.
                    return "Explore", ("explore", "none", "none")
            
            # If door is open or no door exists
            else:
                # Just go to goal
                if goal_pos:
                    return "Go to the goal", ("goto", "green", "goal")
        
        # If we have the key:
        else:
            # If door is closed, open it.
            if door_obj and not door_obj.is_open:
                # Check if key color matches door color?
                # In MiniGrid DoorKey, they usually match.
                if carrying.color == door_obj.color:
                    return f"Open the {door_obj.color} door", ("open", door_obj.color, "door")
                else:
                    # Wrong key? Drop it? (Not typical in standard DoorKey)
                    # Maybe find another key.
                    pass
            
            # If door is open, go to goal.
            if goal_pos:
                return "Go to the goal", ("goto", "green", "goal")

        return "Explore", ("explore", "none", "none")

    def get_negative_subgoals(self, correct_tuple):
        """
        Generates incorrect subgoals for the current state.
        """
        action, color, obj = correct_tuple
        negatives = []
        
        # Helper to format text
        def fmt(a, c, o):
            if a == "goto": return f"Go to the {o}" if o == "goal" else f"Go to the {c} {o}"
            if a == "pick": return f"Pick up the {c} {o}"
            if a == "open": return f"Open the {c} {o}"
            return "Explore"

        # 1. Wrong Action
        if action == "pick":
            negatives.append((fmt("open", color, obj), ("open", color, obj))) # Open the key?
            negatives.append((fmt("goto", "green", "goal"), ("goto", "green", "goal"))) # Go to goal (skipping step)
        elif action == "open":
            negatives.append((fmt("pick", color, obj), ("pick", color, obj))) # Pick up the door?
            negatives.append((fmt("goto", "green", "goal"), ("goto", "green", "goal"))) # Go to goal (skipping step)
        elif action == "goto" and obj == "goal":
            # Wrong action?
            negatives.append(("Pick up the goal", ("pick", "green", "goal")))
        
        # 2. Wrong Object / Color
        # If picking yellow key, suggest picking red key (if it doesn't exist)
        if action == "pick":
            other_colors = {"red", "blue", "green", "purple"} - {color}
            for c in list(other_colors)[:2]:
                 negatives.append((fmt("pick", c, obj), ("pick", c, obj)))
        
        if action == "open":
             other_colors = {"red", "blue", "green", "yellow"} - {color}
             for c in list(other_colors)[:2]:
                 negatives.append((fmt("open", c, obj), ("open", c, obj)))

        return negatives
