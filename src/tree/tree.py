from collections import deque
from typing import List

class Tree():
    """Tree data structure for keeping track of decomposed subclaims."""

    def __init__(self, data: str):
        self.data = data
        self.children = []
        self.parent = None 

    def add_child(self, child: 'Tree') -> None:
        child.parent = self
        self.children.append(child)

    def preorder(self, node: 'Tree') -> List[str]:
        return [] if not node else [node.data] + [j for i in node.children for j in self.preorder(i)]

    def get_level(self) -> int:
        """
        Get the depth of the current node from the root
        """
        level = 0
        parent = self.parent
        while parent:
            level += 1
            parent = parent.parent
        return level

    def level_order(self) -> List[List[list]]:
        """
        Gets nodes per level/ depth from the root
        """
        q = deque([self])
        levels = []
        while q:
            level, level_size = [], len(q)
            for _ in range(level_size):
                current = q.popleft()
                level.append(current.data)
                q.extend(current.children)
            if level:
                levels.append(level)
        return levels

    def get_leaves(self) -> List[str]:
        
        """
        Get all leaves of the tree
        """
        if self.children == []:
            return [self.data]
        
        leaves = []
        for child in self.children:
            if not child.children:
                leaves.append(child.data)
            else:
                leaves.extend(child.get_leaves())

        return leaves

    def bfs_search(self, key: str) -> 'Tree':
        """
        Search for a node with the specified key using BFS.
        """
        q = deque([self])
        while q:
            current = q.popleft()
            if current.data == key:
                return current
            q.extend(current.children)

    def print_tree(self) -> None:
        """
        Prints the tree in tree format
        """
        prefix = '  ' * self.get_level() + "|__ " if self.parent else ""
        print(prefix + self.data)
        for child in self.children:
            child.print_tree()    
