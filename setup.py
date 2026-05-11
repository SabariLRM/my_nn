from setuptools import setup, find_packages, Extension

extensions = [
    Extension("neural_scratch.layers", ["neural_scratch/layers.c"]),
    Extension("neural_scratch.activations", ["neural_scratch/activations.c"]),
    Extension("neural_scratch.losses", ["neural_scratch/losses.c"]),
    Extension("neural_scratch.optimizers", ["neural_scratch/optimizers.c"]),
    Extension("neural_scratch.network", ["neural_scratch/network.c"]),
]

setup_kwargs = dict(
    name="neural_scratch",
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
