# To use a consistent encoding
from codecs import open
from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as readme:
    long_description = readme.read()

lambda_name = "serverless-pypi"

# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
lambda_version = "0.0.6"

lambda_description = "An AWS Lambda implementation for the PyPI protocols"

# How mature is this project? Common values are
#   3 - Alpha
#   4 - Beta
#   5 - Production/Stable
lambda_dev_status = "3 - Alpha"

# What does your project relate to?
lambda_keywords = "lambda"

# What is your project's license
lambda_license = "OSI Approved :: Apache Software License"

# Who is the author?
# TODO: Update the author
lambda_author = "Joseph Wortmann"
lambda_author_email = "jwortmann@quinovas.com"

# List run-time dependencies here.  These will be installed by pip when
# your project is installed. For an analysis of "install_requires" vs pip's
# requirements files see:
# https://packaging.python.org/en/latest/requirements.html
lambda_install_requires = [
    "aws-error-utils",
    "backoff",
    "beautifulsoup4",
    "fastapi",
    "mangum",
    "orjson",
    "passlib",
    "pydantic",
    "python-multipart",
    "requests",
]

##############################################
# CHANGES BELOW HERE ARE MADE AT YOUR OWN RISK
##############################################

setup(
    name=lambda_name,
    version=lambda_version,
    description=lambda_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=lambda_author,
    author_email=lambda_author_email,
    # Choose your license
    license=lambda_license,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: " + lambda_dev_status,
        # Pick your license as you wish (should match "license" above)
        "License :: " + lambda_license,
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.9",
    ],
    keywords=lambda_keywords,
    install_requires=[
        x for x in lambda_install_requires if x not in ["boto3", "botocore"]
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    lambda_package="src/lambda_function",
    setup_requires=["lambda-setuptools"],
    python_requires=">=3.9",
)
