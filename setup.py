from setuptools import setup, find_packages, Extension

extensions = [
    Extension("nn_core.layers", ["nn_core/layers.c"]),
    Extension("nn_core.activations", ["nn_core/activations.c"]),
    Extension("nn_core.losses", ["nn_core/losses.c"]),
    Extension("nn_core.optimizers", ["nn_core/optimizers.c"]),
    Extension("nn_core.network", ["nn_core/network.c"]),
]

setup_kwargs = dict(
    name="nn_core",
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
