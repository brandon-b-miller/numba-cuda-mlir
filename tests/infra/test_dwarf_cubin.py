# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end DWARF tests via cuobjdump on compiled cubins.

- default jit: no .debug_line, no .debug_info
- lineinfo=True: .debug_line present
- debug=True: .debug_info present (full DWARF with types and variables)

Tests that need to compare the two codegen paths cross-compile with
``compile_ptx(cc=...)``, assemble with ``ptxas -g -O0`` and read the result
with ``llvm-dwarfdump``, so they need no device of the target architecture.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache

from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler
import pytest


@lru_cache(maxsize=None)
def _find_tool(name: str) -> str | None:
    """Resolve a toolchain binary once; return None if not found.

    The wheel-installed toolchain is tried first: its version tracks the
    libnvvm this package compiles with, so it understands the same set of
    architectures.
    """
    candidates = []
    spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    if spec is not None:
        for location in spec.submodule_search_locations or []:
            candidates.append(os.path.join(location, "bin", name))
    try:
        from numba_cuda_mlir.tools import get_cuda_toolkit_path

        toolkit = get_cuda_toolkit_path()
        if toolkit:
            candidates.append(os.path.join(toolkit, "bin", name))
    except Exception:
        pass
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return shutil.which(name)


def _tool_path(name: str) -> str:
    """Return cached path for a toolchain binary or skip the test."""
    path = _find_tool(name)
    if path is None:
        pytest.skip(f"{name} not found (set CUDA_HOME or ensure {name} is on PATH).")
    return path


def _cuobjdump_path() -> str:
    """Return cached cuobjdump path or skip the test."""
    return _tool_path("cuobjdump")


def _dwarf_dump(cubin: bytes) -> str:
    """Run `cuobjdump -elf` on cubin bytes; return combined stdout+stderr."""
    cuobjdump = _cuobjdump_path()
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        f.flush()
        path = f.name
    try:
        cp = subprocess.run(
            [cuobjdump, "-elf", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if cp.returncode != 0:
            pytest.fail(
                f"cuobjdump exited with {cp.returncode}\nstdout: {cp.stdout}\nstderr: {cp.stderr}"
            )
        return cp.stdout + cp.stderr
    finally:
        os.unlink(path)


def _compile_and_dump_dwarf(kernel, sig=None) -> str:
    """Compile kernel to cubin, run cuobjdump -elf, return combined stdout+stderr."""
    sig = sig or types.void(types.float32[:])
    cubin = compiler.compile_cubin(kernel, sig)
    return _dwarf_dump(cubin)


def k_compile_unit(out: cuda.DeviceNDArray, a, b):
    c = a + b
    out[0] = c


def _dwarfdump_for_cc(cc) -> str:
    """DWARF of `k_compile_unit` built for `cc`, as llvm-dwarfdump text.

    Skips when the available ptxas predates the requested architecture.
    """
    ptxas = _tool_path("ptxas")
    dwarfdump = _tool_path("llvm-dwarfdump")
    arch = f"sm_{cc[0]}{cc[1]}"
    ptx, _ = compiler.compile_ptx(
        k_compile_unit,
        types.void(types.int32[:], types.int32, types.int32),
        debug=True,
        opt=False,
        cc=cc,
    )
    with tempfile.TemporaryDirectory() as workdir:
        ptx_path = os.path.join(workdir, "k.ptx")
        cubin_path = os.path.join(workdir, "k.cubin")
        with open(ptx_path, "w") as f:
            f.write(ptx)
        cp = subprocess.run(
            [ptxas, "-arch", arch, "-g", "-O0", ptx_path, "-o", cubin_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if cp.returncode != 0:
            if "is not defined for option 'gpu-name'" in cp.stderr:
                pytest.skip(f"{ptxas} cannot target {arch}.")
            pytest.fail(f"ptxas exited with {cp.returncode}\nstderr: {cp.stderr}")
        cp = subprocess.run(
            [dwarfdump, "--debug-info", cubin_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if cp.returncode != 0:
            pytest.fail(f"llvm-dwarfdump exited with {cp.returncode}\nstderr: {cp.stderr}")
        return cp.stdout


def _compile_unit_attrs(dump: str) -> dict[str, str]:
    """Attributes of the first DW_TAG_compile_unit in an llvm-dwarfdump output."""
    lines = dump.splitlines()
    for start, line in enumerate(lines):
        if "DW_TAG_compile_unit" in line:
            break
    else:
        pytest.fail(f"no DW_TAG_compile_unit in DWARF dump:\n{dump}")
    attrs = {}
    for line in lines[start + 1 :]:
        if "DW_TAG_" in line or "NULL" in line:
            break
        match = re.match(r"\s*(DW_AT_\w+)\s+\((.*)\)\s*$", line)
        if match:
            attrs[match.group(1)] = match.group(2).strip('"')
    return attrs


@pytest.mark.parametrize("cc", [(7, 5), (12, 0)], ids=["sm_75", "sm_120"])
def test_cubin_compile_unit_describes_source(cc):
    """Both codegen paths name the real producer, language and source file.

    sm_75 goes through the LLVM70 translator and sm_120 through the modern
    one; the compile unit they emit has to agree.
    """
    directory, filename = os.path.split(os.path.abspath(__file__))
    attrs = _compile_unit_attrs(_dwarfdump_for_cc(cc))

    assert attrs.get("DW_AT_producer") == "clang (numba-cuda-mlir)"
    assert attrs.get("DW_AT_language") == "DW_LANG_C_plus_plus"
    assert attrs.get("DW_AT_name") == filename
    assert attrs.get("DW_AT_comp_dir") == directory


def test_cubin_decl_file_resolves_to_source():
    """Every decl_file on the LLVM70 path resolves to the source that defined the kernel.

    A file entry with a "." directory is resolved against DW_AT_comp_dir, so a
    placeholder on either side composes into an unusable "././<name>".
    """
    dump = _dwarfdump_for_cc((7, 5))
    decl_files = set(re.findall(r'DW_AT_decl_file\s+\("(.*?)"\)', dump))

    assert decl_files == {os.path.abspath(__file__)}, (
        f"unexpected decl_file entries: {sorted(decl_files)}"
    )


def test_cubin_gnu_pubnames_on_modern_path():
    """The modern path asks for GNU pubnames, so ptxas emits a name index.

    The LLVM70 path cannot: LLVMDIBuilderCreateCompileUnit is the only compile
    unit entry point LLVM 7's C API exposes and it takes no name-table
    argument, so there is nowhere to request pubnames from.
    """
    attrs = _compile_unit_attrs(_dwarfdump_for_cc((12, 0)))

    assert attrs.get("DW_AT_GNU_pubnames") == "0x01"


def test_cubin_no_line_table():
    """Default jit must not emit a .debug_line section in the cubin."""

    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert ".debug_line" not in out, (
        f"Default jit should not emit .debug_line section; cuobjdump output:\n{out}"
    )


def test_cubin_line_table():
    """With lineinfo=True, the cubin contains a .debug_line section."""

    @cuda.jit(lineinfo=True, opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert ".debug_line" in out, (
        f"cuobjdump output should contain .debug_line section when lineinfo=True; output:\n{out}"
    )


def test_cubin_no_debug_info_default():
    """Default jit must not emit a .debug_info section."""

    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert ".debug_info" not in out, (
        f"Default jit should not emit .debug_info section; cuobjdump output:\n{out}"
    )


def test_cubin_debug_info():
    """With debug=True, the cubin contains a .debug_info section (full DWARF)."""

    @cuda.jit(debug=True, opt=False)
    def k(out: cuda.DeviceNDArray, a, b):
        out[0] = a + b

    out = _compile_and_dump_dwarf(k, types.void(types.int32[:], types.int32, types.int32))
    assert ".debug_info" in out, (
        f"cuobjdump output should contain .debug_info section when debug=True; output:\n{out}"
    )
