from setuptools import find_packages, setup
from pathlib import Path

package_path = __file__
setup(
    name="ablator",
    version="0.1",
    author="Iordanis Fostiropoulos",
    author_email="dev@iordanis.xyz",
    url="https://iordanis.xyz",
    packages=find_packages(),
    description="Model Ablation Tool-Kit",
    python_requires=">3.10",
    long_description=Path(package_path).parent.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.24.1",
        "pandas==2.0.0",
        "scikit-learn==1.2.2",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "tqdm==4.64.1",
        "tensorboard==2.12.1",
        "matplotlib==3.7.1",
        "omegaconf==2.2.3",
        "scipy==1.10.1",
        "setproctitle==1.3.2",
        "ray==2.1.0",
        "pynvml==11.5.0",
        "optuna==3.1.1",
        "tabulate==0.9.0",
        "seaborn==0.12.2",
    ],
    extras_require={
        "dev": [
            "mypy==1.2.0",
            "pytest==7.3.0",
            "black==23.3.0",
            "flake8==6.0.0",
            "pylint==2.17.2",
        ],
    },
)
