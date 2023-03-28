from setuptools import setup

setup(
    name='netective',
    version='0.5',
    py_modules=['netective'],
    entry_points={
        'console_scripts': [
            'netective=netective:main',
        ],
    },
)
