import re

from . import BaseRule


class TocRule(BaseRule):
    """`目次` will start the table of content(TOC)"""

    def __init__(self):
        self.is_in_TOC = False
        self.last_page = None
        self.head = None

    def run(self, current_node, new_node) -> bool:
        if not self.is_in_TOC:
            if current_node.paragraph.normalized_text == '目次' \
                    and new_node.paragraph.is_header():
                current_node.add_child(new_node)
                self.is_in_TOC = True
                self.head = current_node
        else:
            if new_node.paragraph.page_number != self.last_page:
                is_dot_line = re.match(
                    r'.*\d$', new_node.paragraph.normalized_text) is not None

                is_matched_previous_indentation = new_node.paragraph.indentation \
                                                  in {n.paragraph.indentation
                                                      for n in
                                                      self.head.children}

                if is_matched_previous_indentation and is_dot_line:
                    self.is_in_TOC = True
                else:
                    self.is_in_TOC = False
                    self.head.parent.add_child(new_node)
                    return True

        if self.is_in_TOC:
            self.head.add_child(new_node)
            self.last_page = new_node.paragraph.page_number

        return self.is_in_TOC
