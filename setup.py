"""Setup configuration for Vision-Based 3D Measurement & Perception Tool."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from a file."""
    req_file = Path(__file__).parent / filename
    if req_file.exists():
        with open(req_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="vision-3d-perception",
    version="0.1.0",
    author="Mayank Devangkumar Tamakuwala",
    author_email="maytamaku.saidhwar@gmail.com",
    description="ML-driven vision-based 3D measurement and perception system using depth estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Vision-Based-3D-Measurement-Perception-Tool",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Intended Audience :: Developers",
    #     "Intended Audience :: Science/Research",
    #     "Topic :: Scientific/Engineering :: Artificial Intelligence",
    #     "Topic :: Scientific/Engineering :: Image Recognition",
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: Python :: 3.9",
    #     "Programming Language :: Python :: 3.10",
    #     "Programming Language :: Python :: 3.11",
    #     "Programming Language :: Python :: 3.12",
    # ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "ui": read_requirements("requirements-ui.txt"),
    },
    entry_points={
        "console_scripts": [
            "vision3d-single=scripts.run_single_image:main",
            "vision3d-batch=scripts.run_batch:main",
            "vision3d-video=scripts.run_video:main",
            "vision3d-webcam=scripts.run_webcam:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    keywords="depth-estimation computer-vision 3d-measurement perception ar-ml midas pytorch",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Vision-Based-3D-Measurement-Perception-Tool/issues",
        "Source": "https://github.com/yourusername/Vision-Based-3D-Measurement-Perception-Tool",
    },
)
