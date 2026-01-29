# pynatfmri

A Python library for neuroimaging fMRI connectivity analysis.

## Installation

### From source

```bash
git clone https://github.com/jbartolotti/pynatfmri.git
cd pynatfmri
pip install -e .
```

### With development dependencies

```bash
pip install -e ".[dev]"
```

## Usage

```python
import pynatfmri

# Your code here
```

## Development

To run tests:

```bash
pytest
```

To run code quality checks:

```bash
black src tests
flake8 src tests
mypy src
```

## License

MIT License - see LICENSE file for details

## Author

J. Bartolotti
