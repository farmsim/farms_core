#!/usr/bin/env python
""" Setup script """

import numpy as np  # pylint: disable=wrong-import-position
from Cython.Build import cythonize  # pylint: disable=wrong-import-position
from Cython.Compiler import Options  # pylint: disable=wrong-import-position
from setuptools import setup
from setuptools.extension import Extension

# Cython options
DEBUG = False
Options.docstrings = True
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False
Options.clear_to_none = True
Options.annotate = False
Options.fast_fail = False
Options.warning_errors = False
Options.error_on_unknown_names = True
Options.error_on_uninitialized = True
Options.convert_range = True
Options.cache_builtins = True
Options.gcc_branch_hints = True
Options.lookup_module_cpdef = False
Options.embed = None
Options.cimport_from_pyx = False
Options.buffer_max_dims = 8
Options.closure_freelist_size = 8


setup(
    include_dirs=[np.get_include(), 'farms_core'],
    ext_modules=cythonize(
        [
            Extension(
                f'farms_core.{folder}.*',
                sources=[f'farms_core/{folder}/*.pyx'],
                extra_compile_args=['-O3'],  # , '-fopenmp'
                extra_link_args=['-O3']  # , '-fopenmp'
            )
            for folder in ['array', 'sensors', 'model', 'utils']
        ],
        include_path=[np.get_include(), 'farms_core'],
        compiler_directives={
            # Directives
            'binding': False,
            'embedsignature': True,
            'cdivision': True,
            'language_level': 3,
            'infer_types': True,
            'profile': True,
            'wraparound': False,
            'boundscheck': DEBUG,
            'nonecheck': DEBUG,
            'initializedcheck': DEBUG,
            'overflowcheck': DEBUG,
            'overflowcheck.fold': DEBUG,
            'cdivision_warnings': DEBUG,
            'always_allow_keywords': DEBUG,
            'linetrace': DEBUG,
            # Optimisations
            'optimize.use_switch': True,
            'optimize.unpack_method_calls': True,
            # Warnings
            'warn.undeclared': True,
            'warn.unreachable': True,
            'warn.maybe_uninitialized': True,
            'warn.unused': True,
            'warn.unused_arg': True,
            'warn.unused_result': True,
            'warn.multiple_declarators': True,
        }
    ),
    scripts=[],
    zip_safe=False,
)
