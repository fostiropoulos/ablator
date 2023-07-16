from setuptools import find_packages, setup
from pathlib import Path

package_path = __file__
setup(
    name="ablator",
    version="0.0.1b2",
    author="Iordanis Fostiropoulos",
    author_email="mail@iordanis.me",
    url="https://ablator.org",
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
        "tensorboardX==2.6",
        "matplotlib==3.7.1",
        "omegaconf==2.2.3",
        "scipy==1.10.1",
        "setproctitle==1.3.2",
        # ray default as we need to use the state API
        # keeping it fixed as the API changes a lot between versions.
        "ray[default]==2.5.1",
        # TODO when to migrate from "pydantic<2" to >2 https://github.com/ray-project/ray/issues/37019
        "pydantic==1.10.11",
        "pynvml==11.5.0",
        "optuna==3.1.1",
        "tabulate==0.9.0",
        "seaborn==0.12.2",
        "numpydoc==1.5.0",
        "paramiko==3.2.0",
        "gpustat==1.0",  # must stay fixed because https://github.com/ray-project/ray/issues/35384
    ],
    extras_require={
        "dev": [
            "mypy==1.2.0",
            "pytest==7.3.0",
            "black==23.3.0",
            "flake8==6.0.0",
            "pylint==2.17.2",
            "tensorboard==2.12.2",
            "mock==5.0.2",
            "types-tabulate==0.9.0.2",
            "types-paramiko==3.2.0.0",
            "docker==6.1.3",
            "pytest-xdist==3.3.1",  # plug-in for distributed pytests
            "pytest-order==1.1.0",  # ordering is important for test_mp.py
            "pytest-rerunfailures==12.0",  # randomness in tests can require to re-run them a few times
            "pydoclint==0.1.0"
        ],
    },
)
