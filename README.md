# caching-for-ml-models

Caching for ML models.

This code is used in this [blog post]().

## Requirements

- Python 3

## Installation 

The Makefile included with this project contains targets that help to automate several tasks.

To download the source code execute this command:

```bash
git clone https://github.com/schmidtbri/caching-for-ml-models
```

Then create a virtual environment and activate it:

```bash
# go into the project directory
cd caching-for-ml-models

make venv

source venv/bin/activate
```

Install the dependencies:

```bash
make dependencies
```

## Running the Unit Tests

To run the unit test suite execute these commands:

```bash
# first install the test dependencies
make test-dependencies

# run the test suite
make test

# clean up the unit tests
make clean-test
```
