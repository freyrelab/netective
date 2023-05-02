import pandas as pd
from netective.struct import NormObserver

import pandas as pd


def test_norm_observer():
    # Test with string norm
    norm = "biol"
    norm_observer = NormObserver(norm)
    assert not norm_observer.change()
    assert not norm_observer.change()
    norm_observer.norm = "netheory"
    assert norm_observer.change()
    assert not norm_observer.change()

    # Test with None norm
    norm_observer = NormObserver(None)
    assert not norm_observer.change()
    assert not norm_observer.change()
    norm_observer.norm = "biol"
    assert norm_observer.change()
    assert not norm_observer.change()

    # Test with converting to None
    norm_observer = NormObserver("biol")
    assert not norm_observer.change()
    norm_observer.norm = None
    assert norm_observer.change()
    assert norm_observer.norm is None
    assert not norm_observer.change()

    # Test with pd.Series norm
    data = {"a": 0.8, "b": 0.01, "c": 1}
    series_norm = pd.Series(data)
    norm_observer = NormObserver(series_norm)
    assert not norm_observer.change()
    assert not norm_observer.change()
    update_ser = {"a": 10, "b": 11, "c": 12}
    series_norm = pd.Series(update_ser)
    norm_observer.norm = series_norm
    assert norm_observer.change()
    assert not norm_observer.change()

    # Test with other objects
    norm_observer = NormObserver(42)
    assert not norm_observer.change()
    assert not norm_observer.change()
    norm_observer.norm = 43
    # ignores the change because the norm is not None, a string or a pd.Series
    assert not norm_observer.change()
    assert not norm_observer.change()
