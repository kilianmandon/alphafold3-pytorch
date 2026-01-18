from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

def diff(v1, v2, mask=None, atol=1e-2, rtol=1e-3):
    if mask is not None:
        v1 = v1 * mask
        v2 = v2 * mask
    da = np.abs(v1-v2)
    dr = da / np.clip(np.maximum(np.abs(v1), np.abs(v2)), 1e-10, None)
    diff_mask = (da > atol) &  (dr > rtol)
    di = np.stack(np.nonzero(diff_mask), axis=-1)

    return di, da, dr
    

def broadcast_mask(mask, v):
    mask_inds = set()
    mask_len = len(mask.shape)

    for i in range(len(v.shape) - mask_len + 1):
        if v.shape[i:i+mask_len] == mask.shape:
            mask_inds.add(i)
    
    mask_inds = list(mask_inds)
    if len(mask_inds) == 0:
        raise ValueError(f'Mask shape {mask.shape} not applicable to tensor with shape {v.shape}.')
    elif len(mask_inds) > 1:
        raise ValueError(f'Mask shape {mask.shape} ambiguous for tensor with shape {v.shape}.')

    mask_ind = mask_inds[0]
    mask = mask.reshape((1,) * mask_ind + mask.shape + (1,) * (len(v.shape)-mask_ind-mask_len))
    mask = np.broadcast_to(mask, v.shape)

    return mask


@dataclass
class TensorSpec:
    tensor: Any
    mask: Optional[Any] = None

@dataclass
class DiskTensorSpec:
    name: str
    tensor_files: list[str] = field(default_factory=list)
    mask_files: list[Optional[str]] = field(default_factory=list)
    from_this_session: bool = False
    stack_shape: tuple = ()

_CURRENT_TRACE = None

def current_trace():
    if _CURRENT_TRACE is None:
        raise RuntimeError('No active TensorTrace context.')
    return _CURRENT_TRACE


class Chapter:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        current_trace().add_chapter(self.name)

    def __exit__(self, type, value , traceback):
        current_trace().pop_chapter(self.name)


class TensorTrace:
    def __init__(self, path: str, mode: str, framework='numpy'):
        assert mode in ('read', 'write')
        assert framework in ('numpy', 'pytorch', 'jax')

        self.framework = framework

        self.base_path = Path(path)
        self.index_path = self.base_path / 'index.json'
        self.tensor_dir = self.base_path / 'tensors'
        self.mask_dir = self.base_path / 'masks'
        self.chapters = []

        self.specs: dict[str, DiskTensorSpec] = {}
        self.loading_index: dict[str, DiskTensorSpec] = {}

        self.context = {}

    def set_context(self, name, value):
        self.context[name] = value

    def get_context(self, name):
        return self.context[name]

    def add_chapter(self, name: str):
        self.chapters.append(name)
    
    def pop_chapter(self, name: str):
        assert self.chapters[-1] == name
        self.chapters.pop(-1)

    def load_single_tensor(self, name: str, with_mask=False, silent_loop=True):
        i = self.loading_index.get(name, 0)

        if i >= len(self.specs[name].tensor_files):
            if silent_loop:
                i = 0
            else:
                raise IndexError(f'Reached end of tensor {name}')

        self.loading_index[name] = i + 1

        tensor_filename = self.base_path / self.specs[name].tensor_files[i]
        mask_filename = self.specs[name].mask_files[i]

        tensor = np.load(tensor_filename)

        if not with_mask:
            return tensor
        else:
            if mask_filename is not None:
                mask = np.load(self.base_path / mask_filename)
            else:
                mask = None
            
            return tensor, mask

    def load_tensor(self, name: str, with_mask=False, silent_loop=True, load_single=False):
        stack_shape = self.specs[name].stack_shape
        if len(stack_shape)==0 or load_single:
            if with_mask:
                tensor, mask = self.load_single_tensor(name, with_mask, silent_loop)
                tensor = _from_numpy(tensor, self.framework)
                if mask is not None:
                    mask = _from_numpy(mask, self.framework)
                return tensor, mask
            else:
                tensor = self.load_single_tensor(name, with_mask, silent_loop)
                return _from_numpy(tensor, self.framework)

        n = math.prod(stack_shape)
        tensors = []
        masks = []
        for i in range(n):
            res = self.load_single_tensor(name, with_mask, silent_loop)
            if with_mask:
                tensors.append(res[0])
                masks.append(res[1])
            else:
                tensors.append(res)
        
        tensors = np.stack(tensors, axis=0).reshape(stack_shape + list(tensors[0].shape))
        tensors = _from_numpy(tensors, self.framework)

        if not with_mask:
            return tensors

        if masks[0] is None:
            return tensors, None
        
        masks = np.stack(masks, axis=0).reshape(stack_shape + masks[0].shape)
        masks = _from_numpy(masks, self.framework)
        return tensors, masks

        

    
        

    def load_data(self):
        with open(self.index_path, 'r') as f:
            index = json.load(f)

        self.specs = {
            data['name']: DiskTensorSpec(data['name'], data['tensor_files'], data['mask_files'], False, data['stack_shape'])
            for data in index['tensors']
        }


    def save_index(self):
        index = {
            'tensors': [
                {
                    'name': name,
                    'stack_shape': spec.stack_shape,
                    'tensor_files': spec.tensor_files,
                    'mask_files': spec.mask_files,
                }
                for name, spec in self.specs.items()
            ]
        }
        with open(self.index_path, 'w') as f:
            json.dump(index, f, indent=4)

    def __enter__(self):
        global _CURRENT_TRACE
        _CURRENT_TRACE = self
        
        if self.index_path.exists():
            self.load_data()
        else:
            self.tensor_dir.mkdir(parents=True, exist_ok=True)
            self.mask_dir.mkdir(parents=True, exist_ok=True)
    
    def __exit__(self, type, value, traceback):
        print('Beginning exit...')
        global _CURRENT_TRACE
        _CURRENT_TRACE = None
        print('Done')


    def get_base_name(self):
        if len(self.chapters) > 0:
            return '/'.join(self.chapters) + '/'
        else:
            return ''

    def log(self, value: Union[Any, dict[str, Any]], name: str='', mask: Optional[Any]=None, overwrite=True, stack_shape: tuple =(), no_chapters=False):
        entries = _unify_log_format(value, name, mask)

        if no_chapters:
            base_name = ''
        else:
            base_name = self.get_base_name()

        for name, ts in entries.items():
            full_name = base_name + name

            if full_name not in self.specs or (overwrite and not self.specs[full_name].from_this_session):
                self.specs[full_name] = DiskTensorSpec(full_name, from_this_session=True, stack_shape=stack_shape)

            dts = self.specs[full_name]
            n = len(dts.tensor_files)

            tensor_np = _to_numpy(ts.tensor)
            tensor_path = self.tensor_dir / f'{full_name}.{n}.npy'
            tensor_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(tensor_path, tensor_np)
            dts.tensor_files.append(str(tensor_path.relative_to(self.base_path)))
            
            if ts.mask is not None:
                mask_np = _to_numpy(ts.mask)
                mask_path = self.mask_dir / f'{full_name}.{n}.npy'
                mask_path.parent.mkdir(exist_ok=True, parents=True)
                np.save(mask_path, mask_np)
                dts.mask_files.append(str(mask_path.relative_to(self.base_path)))
            else:
                dts.mask_files.append(None)

        self.save_index()


    def load(self, name: str, processing: dict[str, list[Callable]]=None, expand_names=True, with_mask=False, silent_loop=True, load_single=False):
        processing = _unify_processing_format(processing, name)

        chapters_prefix = '/'.join(self.chapters) + '/' if len(self.chapters) > 0 else ''
        name_prefix = f'{name}/'
        full_name = f'{chapters_prefix}{name}'
        
        entries = {
            k.removeprefix(chapters_prefix): self.load_tensor(k, with_mask=with_mask, silent_loop=silent_loop, load_single=load_single) for k in self.specs if k.startswith(full_name)
        }

        entries = _apply_processing(entries, processing, apply_to_mask=True)

        entries = {
            k.removeprefix(name_prefix): v for k, v in entries.items()
        }


        if len(entries) == 0:
            raise KeyError(f'Tensor {name} does not exist.')
        elif len(entries) == 1:
            return list(entries.values())[0]
        else:
            if expand_names:
                return _expand_flat_dict(entries)
            else:
                return entries


    def load_all(self, name: str, processing=dict[str, list[Callable]]):
        processing = _unify_processing_format(processing, name)
        all_entries = []
        while True:
            try:
                entries = self.load(name, processing, expand_names=False, silent_loop=False)
                all_entries.append(entries)
            except IndexError:
                break

        if isinstance(all_entries[0], dict):
            result = {
                k: [entry[k] for entry in all_entries] for k in all_entries[0]
            }
            return _expand_flat_dict(result)
        else:
            return all_entries


    def build_if_absent(self, name: str, builder: Callable, stack_shape=()):
        full_name = '/'.join(self.chapters + [name])
        matching_names = [k for k in self.specs if k.startswith(full_name)]

        if len(matching_names) == 0 or self.specs[matching_names[0]].from_this_session:
            data = builder()
            self.log(data, name, stack_shape=stack_shape)

        return self.load(name)

    def compare(self, value: (Any|dict[str, Any]), name: str, processing:Optional[dict|list|Any]=None, input_processing:Optional[dict|list]=None, use_mask: (dict|bool)=True):
        if not isinstance(value, dict):
            value = { name: value }

        value = _collapse_nested_dict(value)

        if isinstance(use_mask, dict):
            use_mask = _collapse_nested_dict(use_mask)
        
        input_processing = _unify_processing_format(input_processing, name)
        if input_processing is not None:
            value = _apply_processing(value, input_processing)

        target = self.load(name, processing, expand_names=False, with_mask=True)
        if not isinstance(target, dict):
            target = { name: target }

        for k, v1 in value.items():
            use_mask_for_pair = use_mask if isinstance(use_mask, bool) else use_mask.get(k, True)
            if k not in target:
                raise KeyError(f'Object {k} not present in comparison target.')

            v2, mask = target[k]
            v1 = _to_numpy(v1).astype(float)
            v2 = _to_numpy(v2).astype(float)
            if mask is not None and use_mask_for_pair:
                mask = _to_numpy(mask).astype(float)
                mask = broadcast_mask(mask, v2)
                v1 = v1 * mask
                v2 = v2 * mask

            di, da, dr = diff(v1, v2)
            da_max = da.max()
            dr_max = dr.max()
            combined_max = np.minimum(da, 10*dr).max()
            if di.size > 0:
                print(f'Problems with {k}.')
                print(f'Combined max: {combined_max}')
                ...
        
        target_without_mask = {
            k: v[0] for k, v in target.items()
        }
        if self.framework == 'pytorch':
            input_device = list(value.values())[0].device
            target_without_mask = { k: v.to(device=input_device) for k,v in target_without_mask.items()}

        if len(target_without_mask) == 1:
            target_without_mask = list(target_without_mask.values())[0]
        return target_without_mask


            

def _apply_processing(entries: dict[str, Any], processing: dict[str, Any], apply_to_mask=False):
    for k,v in entries.items():
        for proc in sum([v for k_proc,v  in processing.items() if k.startswith(k_proc)], []):
            if isinstance(v, tuple):
                if apply_to_mask and v[1] is not None:
                    v = proc(v[0]), proc(v[1])
                else:
                    v = proc(v[0]), v[1]
            else:
                v = proc(v)
        entries[k] = v
    return entries


def _collapse_nested_dict(data: dict[str, Any], sep='/', prefix=''):
    collapsed_dict = {}
    prefix = '' if prefix == '' else f'{prefix}/'
    for k, v in data.items():
        if isinstance(v, dict):
            for kk, vv in _collapse_nested_dict(v).items():
                collapsed_dict[f'{prefix}{k}{sep}{kk}'] = vv
        else:
            collapsed_dict[f'{prefix}{k}'] = v

    return collapsed_dict

def _expand_flat_dict(data: dict[str, Any], sep='/'):
    expanded_dict = {}
    for k, v in data.items():
        levels = k.split('/')
        cur_dict = expanded_dict

        for level in levels[:-1]:
            if level not in cur_dict:
                cur_dict[level] = {}
            cur_dict = cur_dict[level]

        cur_dict[levels[-1]] = v

    return expanded_dict
            



def _unify_log_format(value: Union[Any, dict[str, Any]], name: str='', mask: Optional[Any]=None):
    if isinstance(value, dict):
        entries = _collapse_nested_dict(value, prefix=name)
    else:
        entries = {
            name: value
        }

    for k, v in entries.items():
        if not isinstance(v, TensorSpec):
            entries[k] = TensorSpec(v, mask)

    return entries

def _unify_processing_format(procs, name):
    if procs is not None:
        if isinstance(procs, dict):
            procs = _collapse_nested_dict(procs)
        else:
            procs = { '': procs }
        
        procs = {
            name: process if isinstance(process, list) else [process] 
                for name, process in procs.items()
        }
    else:
        procs = {}
    return procs

def log(value: Union[Any, dict[str, Any]], name: str='', mask: Optional[Any]=None, overwrite=True, stack_shape=(), no_chapters=False):
    current_trace().log(value, name, mask, overwrite, stack_shape, no_chapters=no_chapters)

def jax_debug_log(value: Any, name: str='', mask: Optional[Any]=None, stack_shape=()):
    full_name = current_trace().get_base_name() + name
    import jax
    jax.debug.callback(lambda v, m: log(v, full_name, m, stack_shape=stack_shape, no_chapters=True), value, mask, ordered=True)

def build_if_absent(name: str, builder: Callable, stack_shape=()):
    return current_trace().build_if_absent(name, builder, stack_shape)

def load(name: str, processing:Optional[dict|list|Callable]=None):
    return current_trace().load(name, processing)

def load_all(name: str='', processing: Optional[dict|list|Callable]=None):
    return current_trace().load_all(name, processing)

def compare(value: (Any|dict[str, Any]), name: str, processing:Optional[dict|list|Any]=None, input_processing:Optional[dict|list]=None, use_mask: (dict|bool)=True):
    return current_trace().compare(value, name, processing, input_processing, use_mask)

def set_context(name, value):
    current_trace().set_context(name, value)

def get_context(name):
    return current_trace().get_context(name)





def _to_numpy(x):
    try: 
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass

    try: 
        import jax.numpy as jnp
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except ImportError:
        pass

    if isinstance(x, np.ndarray):
        return x

    raise TypeError(f'Unsupported tensor type: {type(x)}')


def _from_numpy(x, framework):
    match framework:
        case 'numpy':
            return x
        case 'pytorch':
            import torch
            return torch.tensor(x)
        case 'jax':
            import jax.numpy as jnp
            return jnp.asarray(x)
        case _:
            raise ValueError(f'Framework not available: {framework}')
