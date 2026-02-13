import pytest
import torch
from backend.utils import nested_compute_size

def test_single_tensor():
    t = torch.zeros(10)
    # size 10 * 4 bytes = 40
    assert nested_compute_size(t, 4) == 40
    # size 10 * 2 bytes = 20
    assert nested_compute_size(t, 2) == 20

def test_list_tuple():
    t1 = torch.zeros(10)
    t2 = torch.zeros(5)

    l = [t1, t2] # 10 + 5 = 15 elements
    assert nested_compute_size(l, 4) == 60

    tup = (t1, t2)
    assert nested_compute_size(tup, 4) == 60

def test_dict():
    t1 = torch.zeros(10)
    d = {'a': t1, 'b': t1} # 20 elements
    assert nested_compute_size(d, 4) == 80

def test_nested_structure():
    t1 = torch.zeros(10)
    # {'x': [t1, (t1, {'y': t1})]} -> 3 * 10 = 30 elements
    obj = {'x': [t1, (t1, {'y': t1})]}
    assert nested_compute_size(obj, 4) == 120

def test_empty_structures():
    assert nested_compute_size({}, 4) == 0
    assert nested_compute_size([], 4) == 0
    assert nested_compute_size((), 4) == 0

def test_ignored_types():
    assert nested_compute_size(123, 4) == 0
    assert nested_compute_size("string", 4) == 0
    assert nested_compute_size(None, 4) == 0

    # Mixed with ignored
    t1 = torch.zeros(10)
    l = [t1, 123, "string"]
    assert nested_compute_size(l, 4) == 40

def test_element_size_impact():
    t = torch.zeros(10)
    assert nested_compute_size(t, 1) == 10
    assert nested_compute_size(t, 8) == 80
