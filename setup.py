from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "dask",
    "distributed",
    "imaxt-image",
    "zarr",
    "numpy<1.20",
    "cython",
    "scipy>=1.5",
    "voluptuous",
    "xarray",
    "opencv-python",
    "owl-pipeline-develop",
    "scikit-image>=0.18",
    "tensorflow",
]

setup_requirements = ["pytest-runner", "flake8"]

test_requirements = ["coverage", "pytest", "pytest-cov", "pytest-mock"]

setup(
    author="Carlos Gonzalez",
    author_email="cgonzal@ast.cam.ac.uk",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="STPT pipeline.",
    entry_points={"owl.pipelines": "stptdev = stpt_pipeline"},
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    data_files=[("conf", ["conf/stpt_pipeline.yaml"])],
    keywords="stpt_pipeline",
    name="stpt_pipeline",
    packages=find_packages(include=["stpt_pipeline"]),
    ext_modules=cythonize(
        Extension("stpt_pipeline.utils", ["stpt_pipeline/utils.pyx"])
    ),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.ast.cam.ac.uk/imaxt/stpt_pipeline",
    version="0.9.1",
    zip_safe=False,
    dependency_links=[
        "https://imaxt.ast.cam.ac.uk/pip/imaxt-image",
        "https://imaxt.ast.cam.ac.uk/pip/owl-pipeline-develop",
    ],
)
