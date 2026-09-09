# PySRF source

This directory contains a modified vendored copy of
[PySRF](https://github.com/florianmahner/pysrf) by Florian P. Mahner. Vendoring
preserves the implementation and dependency compatibility used for the
reported analyses. The included source snapshot was distributed under the MIT
license; see `LICENSE`.

Build the accelerated Cython backend from the repository root after activating
the analysis environment:

```bash
python functions/srf/build_extension.py
```

The command compiles `functions.srf._bsum` in place. Generated C, build, and
shared-library files are ignored by Git.
