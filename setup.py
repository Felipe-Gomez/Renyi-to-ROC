from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class OptionalBuildExt(build_ext):
    """Allow building C extensions to fail gracefully."""
    def run(self):
        try:
            super().run()
        except Exception as e:
            print("WARNING: optional Cython extension build failed, using pure-Python fallback.")
            print(f"  {e}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"WARNING: building extension {ext.name} failed, skipping.")
            print(f"  {e}")

def get_extensions():
    try:
        from Cython.Build import cythonize
        import numpy as np
    except Exception:
        return []

    ext = Extension(
        name="renyi_to_roc._converter",
        sources=["src/renyi_to_roc/cython_converter.pyx"],
        include_dirs=[np.get_include()],
    )

    return cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        annotate=False,
    )

setup(
    package_dir={"": "src"},
    packages=["renyi_to_roc"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
)

