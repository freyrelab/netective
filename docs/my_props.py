# To be able to include your own properties, you need to import the parent class first.
import networkx as nx
from netective.structure.properties import _Property, return_scalar, use_selfloops, check_raw_value, NormalizationError

# Then, you can create your own class inheriting from _Property.
# and implement the corresponding methods. You may want to use this class as a template.
# The decorators are used to define the required preprocessing for the input graph. Import them from netective.structure.properties too.
# Optionally, you may also want to import NormalizationError to raise an error if the property cannot be normalized.
@return_scalar
@use_selfloops
class MyProperty(_Property):
    """MyProperty class to use as a template for custom properties.

    Methods:
        compute: Compute the number of nodes of a graph.
        norm_biol: Not implemented.
        norm_network: Not implemented.
    """

    CLASS_NAME = "My Property" # Human readable name of the property.

    def __init__(self, G: nx.Graph):
        super().__init__(G)

    def compute(self) -> int:
        """Compute the number of nodes.

        Returns:
            int: always return 101.
        """
        self._raw_value = 101
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError(
            "Not implemented for this example."
        )

    @check_raw_value
    def norm_network(self) -> float:
        raise NormalizationError(
            "Not implemented for this example."
        )
