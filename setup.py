"""Setup script for the group activity recognition package."""

from setuptools import setup, find_packages


setup(
    name="deep-activity-recognition",
    version="1.0.0",
    description="Hierarchical Deep Temporal Models for Group Activity Recognition.",
    author="Ad7amstein",
    author_email="adham32003200@gmail.com",
    url="https://github.com/Ad7amstein/deep-activity-recognition",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [],
    },
)
