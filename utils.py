
class CircularStack(list):

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.stack = []

    def update(self, item):
        if len(self.stack) > self.size-1:
            self.stack.pop(0)
        self.stack.append(item)

    def is_full(self):
        return len(self.stack) == self.size

    def __getitem__(self, item):
        return self.stack[item]

    def __iter__(self):
        return iter(self.stack)

    def __len__(self):
        return len(self.stack)