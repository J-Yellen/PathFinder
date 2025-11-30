# Changelog

All notable changes to PathFinder will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes.

## [1.0.0] - 2025-11-30

**🎉 Major Release: Production-Ready Feature Selection Library**

This release marks PathFinder as production-ready with comprehensive enhancements to visualization, testing, API stability, and bug fixes. The library now provides 95% test coverage, enhanced plotting capabilities, and a stable public API.

### Added
- `highlight_top_path` parameter to `plot()` function for visually highlighting the best path
  - Highlights rows corresponding to top path with semi-transparent green overlay
  - Highlighting stops at diagonal (lower triangle only)
  - Automatically colors corresponding axis labels dark green and bold
- `axis_labels` parameter to `plot()` function for automatic label detection from BAM
  - When `axis_labels=True`, uses labels from `BinaryAcceptance.labels` 
  - Respects BAM sorting order (works correctly with WHDFS auto_sort)
  - `xy_labels` parameter still works as override
- `Results.from_list_of_results()` classmethod for creating Results from Result objects
- `Results.from_list_of_dicts()` classmethod for creating Results from list of dictionaries
- `WHDFS.get_sorted_results()` method for accessing Results in sorted index space
- `WHDFS.__str__()` override for user-friendly display in original indices
- `plot_sorted` parameter to `plot()` function for visualising sorted vs original index space
- Comprehensive test coverage for `dfs.py` with 18 new tests covering:
  - `get_sorted_results()` and `get_sorted_paths()` methods
  - String representation with/without auto_sort remapping
  - Property setters (weight_func, wlimit_func, set_top_weight)
  - Default weight threshold behavior
  - Static methods (HDFS.chunked)
  - Path pruning and traversal edge cases
  - Index remapping with/without explicit index_map
  - Type conversions and target node identification
- Comprehensive test coverage for `plot_results.py` with 4 new edge case tests:
  - Empty results handling with highlight_top_path
  - axis_labels behavior when BAM has no labels
  - Combined highlight_top_path and axis_labels functionality
  - Single-node path highlighting
- `test_paths_dont_land_on_black_squares()` validation test preventing plotting regression
- `test_plot_sorted_parameter()` for verifying correct index space handling
- Support for `top=None` in HDFS/WHDFS/Results (unlimited path collection)
- Comprehensive README.md with theoretical background, examples, and API reference
- Tutorial Jupyter notebook (`notebooks/path_test.ipynb`) demonstrating HDFS/WHDFS usage
- Project moved to src/ layout for PEP 517 compliance
- CHANGELOG.md documenting project history

### Changed
- `WHDFS.get_paths` now automatically remaps to original indices when `auto_sort=True` (default)
- `Results.__eq__()` improved to use `get_paths` property (handles WHDFS remapping automatically)
- `Results.__eq__()` now uses `np.allclose` for floating-point weight comparison (rtol=1e-9, atol=1e-12)
- `_top_weights_default()` returns `float('-inf')` when `top=None` or insufficient paths found
- `plot()` now handles `top=None` gracefully (displays all available paths)
- Axis labels in plots now have improved rotation for readability
- Test coverage increased from 85% to 95% (103 tests passing, up from 83)

### Fixed
- **CRITICAL**: Bug in `WHDFS.__init__` where `self.top_weight` was assigned method reference instead of `self._top_weights`
  - Root cause: Line 180 had `self.top_weight = self._top_weights_default` (shadowed method)
  - Solution: Changed to `self._top_weights = self._top_weights_default`
  - Impact: `set_top_weight()` now works correctly, `top_weight()` properly calls custom functions
- **CRITICAL**: Plotting bug where paths appeared on black squares (invalid edges)
  - Root cause: Displaying sorted BAM matrix with paths in original index space
  - Solution: Detect sorted BAM via `_index_map` and unsort matrix for display when `plot_sorted=False`
  - Ensures paths (in original indices) correctly align with displayed matrix
- `plot()` now properly unsorts labels when displaying unsorted matrix
- `add_results()` signature updated to accept `bam` parameter for API compatibility
- WHDFS pruning threshold logic now correctly handles unlimited path collection (`top=None`)
- Results static method `_to_dict()` extracted for improved code reusability

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
- Test coverage increased from 60% to 85%
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

### Version 1.0.0 Highlights

**PathFinder reaches production maturity with this major milestone release.**

Key achievements:
1. **Enhanced Visualization**: New `highlight_top_path` and `axis_labels` parameters for intuitive result interpretation
2. **Critical Bug Fixes**: Resolved WHDFS initialization bug and plotting alignment issues
3. **Comprehensive Testing**: 95% code coverage with 103 passing tests (up from 60% coverage and 19 tests)
4. **API Stability**: Stable public API with automatic index remapping via `auto_sort=True` (default)
5. **Developer Experience**: High-level `find_best_combinations()` convenience function
6. **Production Quality**: Extensive edge case testing, validation tests, and documentation

**Stability Commitment**: v1.0.0 marks API stability. Future changes will follow semantic versioning with backward compatibility within major versions.

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

### Migrating to 1.0.0 from 0.2.x

**No breaking changes** - all existing 0.2.x code will work without modification.

**New Features Available**:
```python
# Enhanced plotting with visual highlights
from pathfinder import plot_results
fig, ax = plot_results.plot(bam, results, 
                            highlight_top_path=True,  # NEW
                            axis_labels=True)         # NEW

# Simpler workflow with convenience function
results = pf.find_best_combinations(matrix, weights, threshold=0.7)
```

**Bug Fixes Applied Automatically**:
- WHDFS `set_top_weight()` now works correctly
- Plotting always shows paths on valid edges (white squares)
- Index space alignment handled automatically

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
