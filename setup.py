from setuptools import setup, find_packages

setup(
    name="feature-evolution",
    version="0.1.0",
    description="Tracking neural network feature evolution during training",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "einops>=0.7.0",
        "transformer-lens>=1.17.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "huggingface-hub>=0.19.0",
    ],
    python_requires='>=3.9',
)
