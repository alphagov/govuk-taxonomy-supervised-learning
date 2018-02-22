class MockIO:
    def __init__(self):
        self.buffer = ""

    def __enter__(self):
        return self

    def __exit__(self, t, v, tr):
        return self

    def write(self, b):
        self.buffer = self.buffer + b