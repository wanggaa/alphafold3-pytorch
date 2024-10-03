import torch
from torch import Tensor

import inspect

def rebuild_inputdata_by_functions(
    input_data: Tensor,
    func,
    discard='self'
    ):
    data = input_data.copy()
    
    sig = inspect.signature(func)
    function_kwargs = set(sig.parameters)
    function_kwargs.discard('self')
    data_kwargs = set(data.keys())
    
    for kw in function_kwargs.difference(data_kwargs):
        data[kw] = None
    for kw in data_kwargs.difference(function_kwargs):
        del data[kw]
    
    return data
    