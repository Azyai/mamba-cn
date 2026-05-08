import ast
import re
from pathlib import Path

from setuptools import find_packages, setup


def get_package_version() -> str:
    init_py = Path(__file__).parent / "mamba_ssm" / "__init__.py"
    match = re.search(
        r"^__version__\s*=\s*(.*)$", init_py.read_text(encoding="utf-8"), re.MULTILINE
    )
    if match is None:
        raise RuntimeError("Unable to find __version__ in mamba_ssm/__init__.py")
    return str(ast.literal_eval(match.group(1)))


setup(
    name="mamba_ssm",
    version=get_package_version(),
    packages=find_packages(),
    python_requires=">=3.9",
    description="Mamba-2 selective state space model (minimal)",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "torch",
        "triton",
        "einops",
        "numpy",
        "packaging",
        "huggingface_hub",
    ],
    extras_require={
        "causal-conv1d": ["causal-conv1d>=1.2.0"],
        "train": ["safetensors", "sentencepiece", "transformers", "tqdm"],
        "rag": ["jieba", "rank-bm25", "sentence-transformers", "faiss-cpu", "pyahocorasick"],
        "agent": ["langchain-openai"],
    },
)
