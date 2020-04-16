"""Oracle classes."""

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Oracle:
    def __init__(self):
        """Defines the polytope."""
        pass

    def label_point(self, point):
        """Labels a point."""
        pass


class ExampleOracle(Oracle):
    def __init__(self):
        """Defines the polytope."""
        super().__init__()

        # Example 2D polygon
        self.polygon = Polygon([(0.2, 0.3), (0.5, 0.1), (0.9, 0.4),
                                (0.7, 0.9), (0.4, 0.7)])

    def label_point(self, point):
        """Labels a point."""
        return self.polygon.contains(Point(point))
