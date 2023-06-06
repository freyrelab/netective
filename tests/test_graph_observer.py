import pytest

import netbiol3 as nb

a = nb.Abasy(rest=False)
from netective.structure import GraphObserver


def test_graph_observer():
    """Test the GraphObserver class."""

    # Test the GraphObserver class with a simple graph.
    G = a.regnet("511145_v2003_sRDB01")
    graph_observer = GraphObserver(G)
    assert not graph_observer.changed()
    assert not graph_observer.changed()

    G.remove_node("crp")
    assert graph_observer.changed()
    assert not graph_observer.changed()

    G = a.regnet("511145_v2003_sRDB01")
    graph_observer = GraphObserver(G)  # data=False by default
    G.nodes["crp"]["NDA"] = "Not DNA class anymore!!!"
    assert not graph_observer.changed()
    assert not graph_observer.changed()

    G = a.regnet("511145_v2003_sRDB01")
    graph_observer = GraphObserver(G, data=True)
    G.nodes["crp"]["NDA"] = "Not DNA class anymore!!!"
    assert graph_observer.changed()
    assert not graph_observer.changed()

    # Test the GraphObserver class when passing a new graph.
    G = a.regnet("511145_v2003_sRDB01")
    graph_observer = GraphObserver(G)
    G = G.copy()  # not the same object
    G.remove_node("crp")
    assert not graph_observer.changed()
    assert not graph_observer.changed()

    G = a.regnet("511145_v2003_sRDB01")
    graph_observer = GraphObserver(G)
    G = G.copy()  # not the same object
    G.remove_node("crp")
    assert graph_observer.changed(G)  # update_G=False
    assert not graph_observer.changed()  # still looks at the original graph
    assert (
        "crp" in graph_observer.G.nodes()
    )  # still looks at the original graph

    G = a.regnet("511145_v2003_sRDB01")
    graph_observer = GraphObserver(G)
    G = G.copy()  # not the same object
    G.remove_node("crp")
    assert graph_observer.changed(G, update_G=True)
    assert not graph_observer.changed()
    assert "crp" not in graph_observer.G.nodes()

    G = a.regnet("511145_v2003_sRDB01")
    graph_observer = GraphObserver(G)
    G = G.copy()  # not the same object
    G.remove_node("crp")
    with pytest.raises(ValueError):
        graph_observer.changed(update_G=True)  # inconsistent
    assert not graph_observer.changed()
    assert "crp" in graph_observer.G.nodes()
