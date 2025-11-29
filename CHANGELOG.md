# Changelog

All notable changes to PathFinder will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README.md with theoretical background, examples, and API reference
- Tutorial Jupyter notebook (`notebooks/path_test.ipynb`) demonstrating HDFS/WHDFS usage
- Project moved to src/ layout for PEP 517 compliance
- CHANGELOG.md documenting project history

## [0.2.0] - 2025-11-29

### Added
- Complete docstrings with Args/Returns/Raises sections across entire codebase
- Comprehensive type hints using `typing` module (List, Dict, Tuple, Sequence, Optional, Union)
- Enhanced numpy type aliases with proper dtype and shape annotations
- `.github/copilot-instructions.md` for AI agent guidance
- IMPROVEMENTS_SUMMARY.md documenting code quality improvements

### Changed
- **Breaking**: Migrated from `setup.py`/`setup.cfg` to modern `pyproject.toml`
- Moved package from root to `src/pathfinder/` for PEP 517 compliance
- Updated to setuptools_scm for automatic git-based versioning
- Improved matplotlib API compatibility (`plt.cm.get_cmap` → `plt.colormaps.get_cmap`)
- Enhanced error messages with better formatting
- Updated test coverage paths to reference `src/pathfinder`
- Enforced British English spelling conventions (visualisation, optimisation, etc.)

### Fixed
- **Critical**: Bug in `result.py` `remap_path()` returning sets instead of lists to `from_dict()`
- Missing `ListedColormap` import in `plot_results.py`
- Deprecated matplotlib API usage
- Test logic errors in `test_plot_results.py` (& → and, numpy array comparisons)
- Dictionary key inconsistencies ('weights' → 'weight')
- Type issues in test suite (float weights, correct path sets)
- Edge tuple unpacking in `test_matrix_handler.py`

### Improved
- Test coverage increased from 60% to 85.33%
- All 19/19 tests now passing
- Performance optimisations in WHDFS early termination
- Documentation quality with examples for complex functions

## [0.1.0] - 2024-12-01

### Added
- Initial public release
- Core HDFS (Hereditary Depth-First Search) algorithm
- WHDFS (Weighted HDFS) with early termination optimisation
- `BinaryAcceptance` class for matrix handling
- `Results` class for path/weight management
- Visualisation module (`plot_results.py`) with matplotlib
- Basic test suite with pytest
- Example usage in `path_test.ipynb`

### Features
- Binary Acceptance Matrix (BAM) construction from correlation/relation matrices
- Efficient combinatorial search avoiding invalid combinations
- Weight-based optimisation for feature selection
- Subset filtering option (`allow_subset` parameter)
- Index remapping for sorted/unsorted feature orderings
- Directed acyclic graph (DAG) representation
- Hereditary condition enforcement

### Implementation
- Graph class for weighted directed graph representation
- Adjacency list implementation
- Dummy target node for path completion
- Bisect insertion for sorted results
- Support for negative weights (with warnings)
- Configurable coverage threshold (60%)

## [0.0.1] - Development

### Added
- Initial algorithm development
- Basic graph structure
- Path-finding logic
- Result storage
- Development workflow established

---

## Release Notes

### Version 0.2.0 Highlights

This release represents a major code quality improvement focused on:

1. **Documentation**: Every function and class now has comprehensive docstrings following Google style with Args/Returns/Raises sections
2. **Type Safety**: Complete type annotations using Python's typing module throughout the codebase
3. **Modern Packaging**: Migration to pyproject.toml with automatic versioning via setuptools_scm
4. **Test Quality**: 85% code coverage with all tests passing
5. **Project Structure**: Adoption of src/ layout following Python packaging best practices

The code is now production-ready with robust documentation, comprehensive testing, and modern Python packaging standards.

### Version 0.1.0 Highlights

Initial release implementing the core PathFinder algorithms:

- **HDFS**: Finds all valid feature combinations respecting pairwise constraints
- **WHDFS**: Optimised variant for weighted feature selection with early termination
- **Performance**: Near O(n log n) complexity for typical use cases vs O(2^n) brute force
- **Flexibility**: Supports custom pairwise metrics, weights, and constraint thresholds

Developed for particle physics analysis combination at ATLAS/CMS, but applicable to any feature selection problem with pairwise constraints.

---

## Migration Guides

### Migrating from 0.1.x to 0.2.x

**Package Location Change**:
```python
# Old (0.1.x)
import pathfinder

# New (0.2.x) - same import, different internal structure
import pathfinder
```

No API changes required - the import statement remains the same, but the package is now installed from `src/pathfinder/`.

**Installation Changes**:
```bash
# Old
python setup.py install

# New (recommended)
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

**Version Access**:
```python
# New in 0.2.x - automatic versioning
import pathfinder
print(pathfinder.__version__)  # e.g., "0.2.0.dev5+g7082425"
```

**Testing**:
```bash
# Old
python -m pytest

# New - configured in pyproject.toml
pytest
```

All existing code using PathFinder 0.1.x APIs will work without modification in 0.2.x.

---

## Deprecation Notices

### Current Deprecations

None currently.

### Future Deprecations (planned for 1.0.0)

- Support for Python 3.9 may be dropped in favour of 3.10+ (to be decided)
- Legacy matplotlib compatibility may require matplotlib >= 3.6

---

## Development Roadmap

### Planned for 0.3.0
- [ ] Parallel processing support for multi-core systems
- [ ] GPU acceleration for large matrices (CuPy integration)
- [ ] Sparse matrix support for very large problems
- [ ] Additional correlation metrics (MIC, distance correlation)
- [ ] Interactive visualisation (Plotly backend)
- [ ] Command-line interface (CLI)

### Planned for 1.0.0
- [ ] Stable API freeze
- [ ] Complete documentation website
- [ ] Performance benchmarking suite
- [ ] Extended tutorial collection
- [ ] Support for streaming/incremental updates
- [ ] Integration with scikit-learn feature selection API

### Under Consideration
- R package wrapper
- Julia implementation
- WebAssembly compilation for browser use
- Distributed computing support (Dask, Ray)
- Approximate algorithms for very large n (>10,000 features)

---

## Acknowledgements

Development supported by:
- University of Glasgow SUPA School of Physics and Astronomy
- ATLAS Collaboration at CERN
- CMS Collaboration at CERN
- Science and Technology Facilities Council (STFC)

Based on research described in:
> Yellen, J. (2025). Search strategies for Beyond Standard Model physics at the LHC.  
> PhD Thesis, University of Glasgow. DOI: 10.5525/gla.thesis.85009

---

## Contact & Support

- **Issues**: https://github.com/J-Yellen/PathFinder/issues
- **Discussions**: https://github.com/J-Yellen/PathFinder/discussions
- **Email**: jamie.yellen@glasgow.ac.uk

For bug reports, please include:
1. PathFinder version (`pathfinder.__version__`)
2. Python version
3. Operating system
4. Minimal reproducible example
5. Error message/traceback

For feature requests, please describe:
1. Use case and motivation
2. Proposed API (if applicable)
3. Expected behaviour
4. Alternative approaches considered
