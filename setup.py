import setuptools
import setuptools.command.develop
import setuptools.command.install

class DevelopCmd(setuptools.command.develop.develop):
    def run(self):
        super().run()
        import juliacall

class InstallCmd(setuptools.command.install.install):
    def run(self):
        super().run()
        import juliacall

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="juliacall",
    version="0.1.0",
    author="Christopher Rowley",
    description="Julia-Python interoperability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cjdoris/PythonCall.jl",
    packages=["juliacall"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
    install_requires=['jill>=0.9.7'],
    cmdclass={'install': InstallCmd, 'develop': DevelopCmd},
)
