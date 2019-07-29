from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'dask',
    'distributed',
    'imaxt-image',
    'zarr',
    'numpy',
    'scipy',
    'voluptuous',
]

setup_requirements = ['pytest-runner', 'flake8']

test_requirements = ['coverage', 'pytest', 'pytest-cov', 'pytest-mock']

setup(
    author='Carlos Gonzalez',
    author_email='cgonzal@ast.cam.ac.uk',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description='STPT pipeline.',
    entry_points={'owl.pipelines': 'stpt = stpt_pipeline'},
    install_requires=requirements,
    license='GNU General Public License v3',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    data_files=[('conf', ['conf/stpt_pipeline.yaml'])],
    keywords='stpt_pipeline',
    name='stpt_pipeline',
    packages=find_packages(include=['stpt_pipeline']),
    ext_modules=cythonize(
        Extension('stpt_pipeline.utils', ['stpt_pipeline/utils.pyx'])
    ),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.ast.cam.ac.uk/imaxt/stpt_pipeline',
    version='0.1.0',
    zip_safe=False,
)
