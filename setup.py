import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="radiogalaxies_bnns",
    version="0.0.1",
    author="Devina Mohan",
    author_email="TODO",
    description="Evaluating approximate bayesian inference for radio galaxy classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devinamhn/RadioGalaxies-BNNs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)