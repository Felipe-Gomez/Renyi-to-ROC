try:
    # compiled code (built from cython_converter.pyx)
    from .cython_converter import get_FNR  # if extension overwrote module name
except Exception:
    # fallback to pure python
    from .converter import get_FNR

