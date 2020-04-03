class Oracle:
    def __init__(self):
        pass

    def label_point(self, point):
        pass


class ExampleOracle(Oracle):
    def __init__(self):
        super().__init__()

        # Example 2D polygon
        self.coords = [[0.3, 0.7], [0.3, 0.7]]

    def label_point(self, point):
        arr = [coord1 <= point[i] <= coord2
               for i, (coord1, coord2) in enumerate(self.coords)]

        if all(arr):
            return True
        return False
