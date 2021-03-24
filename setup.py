import setuptools

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
)
