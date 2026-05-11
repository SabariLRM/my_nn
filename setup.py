from setuptools import setup, find_packages, Extension

extensions = [
    Extension("my_nn.layers", ["my_nn/layers.c"]),
    Extension("my_nn.activations", ["my_nn/activations.c"]),
    Extension("my_nn.losses", ["my_nn/losses.c"]),
    Extension("my_nn.optimizers", ["my_nn/optimizers.c"]),
    Extension("my_nn.network", ["my_nn/network.c"]),
]

setup_kwargs = dict(
    name="my_nn",
    version="0.1.0",
    description="A tiny neural network library built from scratch with NumPy.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=extensions,
)

setup(**setup_kwargs)
