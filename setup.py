from distutils.core import setup

with open("requirements.txt") as f:
    requirements = []
    for line in f:
        if not line.startswith(("-e", "-r", "git+", "http://", "https://")):
            requirements.append(line.strip())

setup(
    name="orpo",
    packages=[
        "occupancy_measures",
        "occupancy_measures.agents",
        "occupancy_measures.envs",
        "occupancy_measures.experiments",
        "occupancy_measures.models",
        "occupancy_measures.utils",
    ],
    package_data={"occupancy_measures": ["py.typed"]},
    version="0.0.1",
    license="MIT",
    description='Code for "Preventing Reward Hacking using Occupancy Measure Regularization"',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Cassidy Laidlaw",
    author_email="cassidy_laidlaw@berkeley.edu",
    url="https://github.com/cassidylaidlaw/orpo",
    keywords=[
        "machine learning",
        "reinforcement learning",
        "AI safety",
        "reward misspecification",
    ],
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
