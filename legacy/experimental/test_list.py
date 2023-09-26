
import pandas as pd
import pytest

import hashlib
import pandas as pd

class NormObserver:
    """A class to observe changes in the normalization strategy."""
    def __init__(self, norm):
        """
        Initialize the NormObserver class.
        
        Args:
            norm (None | str | pd.Series): The norm to observe.

        """
        self.norm = norm
        self.norm_hash = self._compute_hash()

    def _hash(self, str_norm):
        """
        Compute the SHA-1 hash of a string.
        Implemented to DRY the code.

        Args:
            str_norm (str): The string to compute the hash.

        Returns:
            str: The hash of the string.
        """
        hash_object = hashlib.sha1(str_norm.encode('utf-8'))
        return hash_object.hexdigest()

    def _compute_hash(self):
        """
        Compute the SHA-1 hash of the current norm value
        
        Args:
            norm (None | str | pd.Series): The norm to compute the hash.
        
        Returns:
            str: The hash of the norm. If norm is None, return None.
        """
        if self.norm is None:
            return None
        
        elif isinstance(self.norm, str):
            str_norm = self.norm
            return self._hash(str_norm)

        elif isinstance(self.norm, pd.Series):
            # convert it to a flat string to be hashed
            # self.norm.to_string(index=True, dtype=True, name=True, length=True, header=True)
            str_norm = f"pd.Series: {self.norm}"
            return self._hash(str_norm)

    def change(self):
        """
        Check if norm has changed with reference to the last call.

        Returns:
            bool: True if the norm has changed, False otherwise.
        """
        new_hash = self._compute_hash()
        if new_hash != self.norm_hash:
            change_flag = True
            self.norm_hash = new_hash
        else:
            change_flag = False
        return change_flag


class TestNormObserver:
    @pytest.fixture
    def norm_observer(self):
        return NormObserver(None)
    
    @pytest.fixture
    def series_norm():
        data = {'a': 0.8, 'b': 0.01, 'c': 1}
        return pd.Series(data)
    
    def test_change_with_string_norm(self, norm_observer):
        norm_observer.norm = "biol"
        assert norm_observer.change() is True
        assert norm_observer.change() is False
        
        norm_observer.norm = "netheory"
        assert norm_observer.change() is True
        assert norm_observer.change() is False
        
    def test_change_with_none_norm(self, norm_observer):
        norm_observer.norm = None
        assert norm_observer.change() is True
        assert norm_observer.change() is False
        
        norm_observer.norm = "biol"
        assert norm_observer.change() is True
        assert norm_observer.change() is False
        
    def test_change_with_pd_series_norm(self, series_norm):
        norm_observer.norm = series_norm
        assert norm_observer.change() is True
        assert norm_observer.change() is False

        norm_observer.norm = None
        assert norm_observer.change() is True
        assert norm_observer.change() is False

        norm_observer = NormObserver(series_norm)
        assert norm_observer.change() is False
        update_ser = {'a': 10, 'b': 11, 'c': 12}
        norm_observer.norm = pd.Series(update_ser)
        assert norm_observer.change() is True
        assert norm_observer.change() is False
        
    def test_change_with_other_objects(norm_observer):
        norm_observer.norm = 42 # ignores the change because the norm is not None, a string or a pd.Series
        # TODO: check if this is the desired behavior, validating the input only in the norm_setter (Srucutre)
        assert norm_observer.change() is False
        assert norm_observer.change() is False
