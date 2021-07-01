from setuptools import setup
import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent / 'tests'))

'''
python3 -m unittest
vim setup.py
rm -rf dist/
python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
'''

setup(
    name='cirneco',
    version='0.0.1',
    url='https://github.com/KuramitsuLab/cirneco.git',
    license='MIT',
    author='Kuramitsu Labo',
    description='CirnecoDell`Etna',
    install_requires=['setuptools', 'pegtree'],
    packages=['cirneco'],
    package_data={'cirneco': [
        '*/*', '*/*', '*.tpeg', '*.txt']},
    entry_points={
         'console_scripts': [
             'cirneco = chirneco.cirneco:main'
         ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing',
    ],
    # test_suite='test_all.suite'
)
