from setuptools import setup, find_packages

setup(
    name="csaf-pain-estimation",
    version="1.0.0",
    description=(
        "Cross-Spectral Attention Fusion (CSAF) for objective pain intensity "
        "estimation from synchronized thermal and RGB facial video."
    ),
    author="Oussama El Othmani, Sami Naouali",
    author_email="salnawali@kfu.edu.sa",
    url="https://github.com/oussama123-ai/Cross-Spectral-Fusion-of-Thermal",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "albumentations>=1.3.0",
        "insightface>=0.7.3",
        "PyYAML>=6.0",
        "omegaconf>=2.3.0",
        "einops>=0.6.1",
        "timm>=0.9.2",
        "tqdm>=4.65.0",
        "rich>=13.3.0",
        "pingouin>=0.5.3",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
