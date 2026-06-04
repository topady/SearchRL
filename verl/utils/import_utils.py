# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

from functools import cache
from typing import List


@cache
def is_megatron_core_available():
    try:
        from megatron.core import parallel_state as mpu
        return True
    except ImportError:
        return False


@cache
def is_vllm_available():
    try:
        import vllm
        return True
    except ImportError:
        return False


def import_external_libs(external_libs=None):
    if external_libs is None:
        return
    if not isinstance(external_libs, List):
        external_libs = [external_libs]
    import importlib
    for external_lib in external_libs:
        importlib.import_module(external_lib)


def load_extern_function(path: str, name: str = "compute_score"):
    """Dynamically load a callable from a Python file.

    Args:
        path: file path relative to project root, e.g. 'reward_functions/rubric_merged.py'
        name: function name in the module (default: 'compute_score')

    Returns:
        callable
    """
    import importlib.util
    import os

    if not os.path.isabs(path):
        # resolve relative path against the project root (where setup.py/pyproject.toml lives)
        # search upward from this file to find the project root
        search_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = search_dir
        for _ in range(8):
            if os.path.exists(os.path.join(project_root, 'setup.py')) or \
               os.path.exists(os.path.join(project_root, 'pyproject.toml')):
                break
            parent = os.path.dirname(project_root)
            if parent == project_root:
                break
            project_root = parent
        path = os.path.join(project_root, path)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)
