from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).parent


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def read_requirements() -> list[str]:
    requirements = []
    for line in read_text("requirements.txt").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


subpackages = find_packages(where=".")
packages = ["annotate"] + [f"annotate.{name}" for name in subpackages]


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            item for item in modules
            if not (item[0] == "annotate" and item[1] == "setup")
        ]

setup(
    name="annotate",
    version="0.1.0",
    description="Multi-agent clinical conversation annotation pipeline",
    long_description=read_text("README.md"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=packages,
    package_dir={"annotate": "."},
    include_package_data=True,
    exclude_package_data={"": ["__pycache__/*", "*.pyc", "*.pyo"], "annotate": ["setup.py"]},
    package_data={
        "annotate": [
            "prompts/*.json",
            "prompts/*.txt",
            "data/raw/*.json",
            "data/raw/*.txt",
            "data/sessions/*.json",
        ],
    },
    install_requires=read_requirements(),
    cmdclass={"build_py": build_py},
    entry_points={
        "console_scripts": [
            "annotate-single=annotate.examples.run_single_conversation:main",
            "annotate-batch=annotate.examples.run_batch_pipeline:main",
            "annotate-generate=annotate.examples.generate_dataset:main",
        ],
    },
)
