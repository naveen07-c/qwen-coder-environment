"""Setup script for NetPulse package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="netpulse",
    version="1.0.0",
    author="NetPulse Team",
    description="Intelligent Network Anomaly Detector using Unsupervised ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/netpulse/netpulse",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "netpulse=netpulse.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "netpulse": ["config.toml"],
    },
)
