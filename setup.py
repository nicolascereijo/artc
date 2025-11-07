from setuptools import setup, find_packages


# Read dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requires: list[str] = f.read().splitlines()

# Read long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description: str = f.read()

_ = setup(
    name="artc",
    version="1.0b4",
    description="Beta version of the ARtC (Audio Real-time Comparator) core",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicolás Cereijo Ranchal",
    author_email="nicolascereijo.careers@protonmail.com",
    url="https://github.com/NicolasCereijo/artc",
    keywords=["audio", "analysis", "comparison", "real-time", "data collection"],
    license="MIT",
    python_requires=">=3.10,<3.13",
    install_requires=requires,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "artc.core.configurations": ["default_configurations.json"],
        "artc.cli": ["commands.json"],
    },
    entry_points={
        "console_scripts": [
            "artc=artc.__main__:main",
        ],
    },
)
