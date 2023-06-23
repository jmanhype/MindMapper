class Thought_Tree_Explorer:
    def __init__(self, strategy='depth_first'):
        self.strategy = strategy

    def explore_thought_tree(self, agent):
        if self.strategy == 'depth_first':
            return self.depth_first_search(agent.thought_tree_root)
        elif self.strategy == 'breadth_first':
            return self.breadth_first_search(agent.thought_tree_root)
        elif self.strategy == 'heuristic':
            return self.heuristic_search(agent.thought_tree_root)
        else:
            raise ValueError("Invalid exploration strategy")

    def depth_first_search(self, node):
        if node is None:
            return None

        if node.is_goal():
            return node

        for child in node.children:
            result = self.depth_first_search(child)
            if result is not None:
                return result

        return None

    def breadth_first_search(self, node):
        if node is None:
            return None

        queue = [node]

        while queue:
            current_node = queue.pop(0)

            if current_node.is_goal():
                return current_node

            queue.extend(current_node.children)

        return None

    def heuristic_search(self, node):
        if node is None:
            return None

        open_list = [node]
        closed_list = []

        while open_list:
            current_node = min(open_list, key=lambda x: x.heuristic_value())
            open_list.remove(current_node)
            closed_list.append(current_node)

            if current_node.is_goal():
                return current_node

            for child in current_node.children:
                if child not in closed_list:
                    open_list.append(child)

        return None

    def prune_thought_tree(self, node, max_depth):
        if node is None or max_depth < 0:
            return

        if max_depth == 0:
            node.children = []
        else:
            for child in node.children:
                self.prune_thought_tree(child, max_depth - 1)