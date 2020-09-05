import random


class SkipList:
    class Node:
        def __init__(self, next, down, key, value):
            self.next = next
            self.down = down
            self.key = key
            self.value = value

    def __init__(self, less=lambda a, b: a < b, equal=lambda a, b: a == b):
        self.__head = SkipList.Node(None, None, None, None)
        self.__less = less
        self.__equal = equal
        self.__MAX_LEVEL = 128
        head_node = self.__head
        for _ in range(self.__MAX_LEVEL-1):
            head_node.down = SkipList.Node(None, None, None, None)
            head_node = head_node.down

    def __random_level(self):
        return random.randint(1, self.__MAX_LEVEL)

    def __nodes_for_update(self, key):
        result = []
        node = self.__head
        while True:
            while node.next is not None and self.__less(node.next.key, key):
                node = node.next

            result.append(node)
            if node.down is not None:
                node = node.down
            else:
                break
        return result

    def insert(self, key, value=None):
        update_nodes = self.__nodes_for_update(key)
        if update_nodes[-1].next is not None and self.__equal(update_nodes[-1].next.key, key):
            return
        new_level = self.__random_level()
        down_node = None
        for level in range(len(update_nodes)-1, -1, len(update_nodes)-1-new_level):
            update_nodes[level].next = SkipList.Node(update_nodes[level].next, down_node, key, value)
            down_node = update_nodes[level].next

    def remove(self, key):
        update_nodes = self.__nodes_for_update(key)
        level = len(update_nodes)-1
        exist = False
        while update_nodes[level].next is not None and self.__equal(update_nodes[level].next.key, key):
            exist = True
            update_nodes[level].next = update_nodes[level].next.next
        if not exist:
            raise KeyError(f"KeyError: {key}")

    def search(self, key):
        update_nodes = self.__nodes_for_update(key)
        if update_nodes[-1].next is not None and self.__equal(update_nodes[-1].next.key, key):
            return [True, update_nodes[-1].next.value]
        return [False, None]



