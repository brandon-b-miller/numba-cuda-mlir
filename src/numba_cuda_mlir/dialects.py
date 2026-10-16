# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register an out-of-tree dialect into this package's MLIR.

Loading a dialect plugin is a process-wide act: it decides which MLIR the
plugin binds to, and getting it wrong produces a second MLIR whose types are
silently unrelated to the first. That makes it the runtime owner's job rather
than each dialect's, which is why it lives here and not in the dialects.

The contract a plugin has to satisfy is deliberately small:

``MLIR_DIALECT_LIBRARY``
    Absolute path to a shared object that carries ``DT_NEEDED`` on
    ``libMLIRPythonCAPI.so`` and leaves MLIR's symbols to it.

``MLIR_DIALECT_ENTRY``
    Name of an ``extern "C" void(MlirContext)`` entry point in that object,
    as emitted by ``MLIR_DEFINE_CAPI_DIALECT_REGISTRATION``.

Why this does not promote anything
----------------------------------
An earlier version of this dance re-opened ``libMLIRPythonCAPI.so`` with
``RTLD_GLOBAL`` so that a plugin built with MLIR's symbols left *undefined*
could scavenge them out of the global scope. That works, and it also inserts
~158k ``mlir::`` symbols into the global lookup scope, where they preempt the
same symbols inside any library loaded afterwards that vendors its own MLIR.
It additionally defeats the ``RTLD_DEEPBIND`` preload this package performs for
exactly the opposite reason (see #170).

Linking the plugin against the aggregate instead makes the dependency explicit.
The preload has already mapped the aggregate by absolute path, so the plugin's
``DT_NEEDED`` resolves against *that* copy by soname, nothing is promoted, and
a third party's vendored MLIR is untouched.
"""

from __future__ import annotations

import ctypes

__all__ = ["register_dialect"]


def _entry_point(library: str, symbol: str):
    """Open ``library`` locally and return its registration function."""
    # RTLD_LOCAL on purpose. The plugin's own symbols are of no interest to
    # anyone else, and its MLIR symbols arrive through DT_NEEDED rather than
    # through the global scope.
    try:
        handle = ctypes.CDLL(library, mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        raise OSError(
            f"could not load dialect plugin {library!r}: {error}. "
            "A plugin must link against libMLIRPythonCAPI.so; if it instead "
            "leaves MLIR's symbols undefined it can only be loaded by "
            "polluting the global symbol scope, which this package does not do."
        ) from error

    fn = getattr(handle, symbol)
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = None
    # Keep the handle alive for the life of the process; the context holds
    # pointers into this object once registration has run.
    _loaded.append(handle)
    return fn


_loaded: list = []


def _raw_context_pointer(context) -> int:
    """The ``MlirContext`` inside the bindings' capsule, as a plain pointer.

    ``MlirContext`` is a single-pointer struct, so under the SysV C ABI it is
    passed exactly like a ``void *``.
    """
    capsule = context._CAPIPtr
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    return get_pointer(capsule, get_name(capsule))


def register_dialect(plugin, context=None):
    """Register ``plugin``'s dialect into ``context``.

    Parameters
    ----------
    plugin:
        A module (or any object) exposing ``MLIR_DIALECT_LIBRARY`` and
        ``MLIR_DIALECT_ENTRY``.
    context:
        The MLIR context to register into. Defaults to this package's context,
        which is the one its own lowering runs in -- registering elsewhere
        would produce a dialect this package's passes cannot see.

    Returns
    -------
    The context, so callers can chain.
    """
    try:
        library = plugin.MLIR_DIALECT_LIBRARY
        symbol = plugin.MLIR_DIALECT_ENTRY
    except AttributeError as error:
        raise TypeError(
            f"{plugin!r} is not a dialect plugin: expected attributes "
            "MLIR_DIALECT_LIBRARY and MLIR_DIALECT_ENTRY"
        ) from error

    if context is None:
        from numba_cuda_mlir.lowering_utilities.context import get_context

        context = get_context()

    _entry_point(library, symbol)(_raw_context_pointer(context))
    return context
