class BaseRule:
    def run(self, current_node, new_node) -> bool:
        """ Return True if this rule can find the parent node of the `new_node`, False otherwise.
        :param current_node: The last inserted leaf on the tree, also, the direct previous paragraph of the new node
        :param new_node: The node that needs to be inserted to the tree.
        """
        raise NotImplementedError()

    @property
    def name(self):
        """ :return: Name of the rule, for debugging only. """
        return self.__class__.__name__
