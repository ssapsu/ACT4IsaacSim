from setuptools import setup

package_name = "dataset_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ssapsu",
    maintainer_email="hans324oh@gmail.com",
    description="Record specified topics to dataset files",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "recorder = dataset_recorder.recorder:main",
            "ACTPolicy = dataset_recorder.actpolicy:main",
        ],
    },
)
