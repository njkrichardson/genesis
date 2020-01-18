import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genesis",  # Replace with your own username
    version="0.0.1",
    author="nick richardson and michael wehner",
    author_email="nrichardson@hmc.edu, mcwehner@gmail.com",
    description="Learning, inference, and information: from the ground up",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/njkrichardson/genesis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "black",
        "flake8",
        "isort",
        "jupyterlab~=1.2.5",
        "matplotlib",
        "mypy",
        "numpy~=1.18.1",
        "pandas",
        "scipy",
        "seaborn",
        "sklearn",
    ],
)
