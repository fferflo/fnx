#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="fnx",
    version="0.1.0",
    python_requires=">=3",
    description="Functional Neural Networks with Jax",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    url="https://github.com/fferflo/fnx",
    packages=find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires=[
        "numpy",
        "jax",
        "einx",
        "dm-haiku",
        "tqdm",
        "requests",
        "pyunpack",
        "imageio",
        "tensorflow",
        "torch",
        "gdown",
        "huggingface_hub",
        "tiktoken",
        "transformers",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
