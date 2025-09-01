from setuptools import setup, find_packages
import os

# Get long description from README if it exists
long_description = "Dynamic Linear Models with Discount Factors"
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Define requirements directly in setup.py
requirements = [
    "numpy>=1.20.0",
    "scipy>=1.7.0", 
    "matplotlib>=3.3.0"
]

setup(
    name="dlm_discount",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dynamic Linear Models with Discount Factors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jportizPD/dlm_discount",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)