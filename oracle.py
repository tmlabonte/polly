from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Oracle:
    def __init__(self):
        pass

    def label_point(self, point):
        pass


class ExampleOracle(Oracle):
    def __init__(self):
        super().__init__()

        # Example 2D polygon
        self.polygon = Polygon([(0.3, 0.3), (0.7, 0.3), (0.7, 0.8), (0.3, 0.5)])

    def label_point(self, point):
        return self.polygon.contains(Point(point))
