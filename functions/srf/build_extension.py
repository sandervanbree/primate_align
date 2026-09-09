#!/usr/bin/env python3
"""Build the local PySRF Cython extension in place."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parents[2]
SRF_DIR = ROOT / "functions" / "srf"
BUILD_DIR = SRF_DIR / "build"


def main() -> None:
    os.chdir(ROOT)
    extension = Extension(
        "functions.srf._bsum",
        ["functions/srf/_bsum.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    setup(
        name="primate-align-srf-extension",
        script_args=["build_ext", "--inplace", "--build-temp", str(BUILD_DIR)],
        ext_modules=cythonize(
            [extension],
            build_dir="functions/srf/build/cython",
            compiler_directives={"language_level": "3"},
        ),
    )


if __name__ == "__main__":
    main()
