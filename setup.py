from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jain-mf",
    version="1.0.0",
    author="JAIN-MF Implementation Team",
    author_email="20231401215@xju.edu.cn",
    description="Tri-Modal Action Recognition via Skeleton-Guided Sparse Subspace Alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/JAIN-MF",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
    },
    entry_points={
        "console_scripts": [
            "jain-mf-train=scripts.train:main",
            "jain-mf-test=scripts.test:main",
        ],
    },
)