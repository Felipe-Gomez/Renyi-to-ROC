from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

# Env toggles
NO_COMPILE = os.environ.get("RENYI_TO_ROC_NO_COMPILE") == "1"
REQUIRE_COMPILE = os.environ.get("RENYI_TO_ROC_REQUIRE_COMPILE") == "1"

# If both are set, REQUIRE_COMPILE should win (user explicitly demands compile).
if NO_COMPILE and REQUIRE_COMPILE:
    NO_COMPILE = False


class OptionalBuildExt(build_ext):
    """
    Build C/Cython extensions, but allow failure unless REQUIRE_COMPILE=1.
    Prints clear build-time messages.
    """

    def run(self):
        if NO_COMPILE:
            print("Renyi-to-ROC: RENYI_TO_ROC_NO_COMPILE=1 -> skipping cython compilation; using pure Python.")
            return

        try:
            super().run()
        except Exception as e:
            if REQUIRE_COMPILE:
                # Hard fail: user demanded compilation
                raise
            print("Renyi-to-ROC: Cython compilation failed; falling back to pure Python.")
            print(f"  {e}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
            print(f"Renyi-to-ROC: Cython compiled successfully: {ext.name}")
        except Exception as e:
            if REQUIRE_COMPILE:
                raise
            print(f"Renyi-to-ROC: WARNING: Cython compilation failed; using pure Python: {ext.name}")
            print(f"  {e}")


def get_extensions():
    if NO_COMPILE:
        return []

    try:
        from Cython.Build import cythonize
        import numpy as np
    except Exception as e:
        if REQUIRE_COMPILE:
            raise RuntimeError(
                "Renyi-to-ROC: RENYI_TO_ROC_REQUIRE_COMPILE=1 but build requirements "
                "(Cython/numpy) are not available in the build environment."
            ) from e
        print("Renyi-to-ROC: Cython/numpy not available for build; using pure Python.")
        print(f"  {e}")
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

