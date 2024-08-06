import os
import shutil
import subprocess
from os.path import join

from setuptools import setup
from setuptools.command.build_py import build_py


class build(build_py):
    def run(self):
        lib_name = "why3py"
        # Use dune to build the OCaml library and copy it in src/bin
        proc = subprocess.run(["dune", "build", "--root", "ocaml"])
        assert proc.returncode == 0, "Error building the OCaml library."
        dune_build = join("ocaml", "_build", "default", "bin")
        # Copy the shared library binary
        bin_path = join("src", lib_name, "bin")
        os.makedirs(bin_path, exist_ok=True)
        dll = join(bin_path, "core.so")
        shutil.copy(join(dune_build, "why3py_ocaml.so"), dll)
        os.chmod(dll, 0o666)
        # Generate stub
        generator = join(dune_build, f"why3py_ocaml.exe")
        cmd = [generator, f"generate-py", "--lib-name", lib_name]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE)
        if proc.returncode != 0:
            print(proc.stdout)
            print("Error generating Python stubs.")
        file = join("src", lib_name, f"core.py")
        with open(file, "w") as f:
            f.write(proc.stdout)
        # Format the stub with black
        if shutil.which("black") is not None:
            subprocess.run(["black", file])


setup(cmdclass={"build_py": build}, setup_requires=["black"])
