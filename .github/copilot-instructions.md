# PathFinder - AI Agent Instructions

## Project Overview
PathFinder is a graph algorithm library for identifying optimal subsets of elements based on pairwise relations and weights. The core algorithm is **Hereditary Depth First Search (HDFS)**, which finds all valid subsets where pairwise relations stay below a threshold.

**Primary use case**: Feature selection for ML training, identifying minimally correlated features from correlation matrices.

**Note**: Documentation follows British English spelling conventions.

## Architecture

### Core Components (read these files together to understand the system)
1. **`pathfinder/matrix_handler.py`** - `BinaryAcceptance` class converts correlation/relation matrices to graph representation
   - Takes float matrix + threshold → creates boolean acceptance matrix
   - Builds weighted directed graph with dummy target node (enables path completion)
   - **Key attribute**: `_index_map` stores the mapping when `sort_bam_by_weight()` is called
   - **Key method**: `sort_bam_by_weight()` returns and stores index map for remapping results back to original order

2. **`pathfinder/dfs.py`** - Two search algorithms:
   - `HDFS`: Exhaustive search, finds ALL valid paths (expensive but comprehensive)
   - `WHDFS`: Weighted search, prunes paths early based on weight limits (faster, finds top paths)
   - Both inherit from `Results` and operate on `BinaryAcceptance` objects
   - **Key feature**: `remap_path()` can be called without arguments - automatically uses `self.bam._index_map`
   - `WHDFS.get_paths` automatically remaps to original indices when `auto_sort=True` (default)
   - `WHDFS.get_sorted_paths()` returns paths in sorted index space (for plotting)

3. **`pathfinder/result.py`** - `Results` class manages path collections
   - Stores `Result` dataclass objects (path as set, weight as float)
   - **Critical feature**: `allow_subset=False` filters out paths that are subsets of other paths
   - `remap_path(index_map=None)` reverses the sorting; if `index_map=None`, it's a no-op

4. **`pathfinder/plot_results.py`** - Matplotlib visualization of BAM matrices + overlaid paths
   - Automatically detects `WHDFS` and uses `get_sorted_paths()` to align with sorted BAM display

### Data Flow Pattern
```
Float Matrix → BinaryAcceptance (with threshold) → HDFS/WHDFS.find_paths() → Results → remap_path() → Original indices
                                     ↓ stores _index_map                                      ↑ auto-detects from bam
```

**Important**: `BinaryAcceptance` owns the `_index_map` state. `HDFS`/`WHDFS` reference it via `self.bam._index_map`, enabling `remap_path()` to work without explicit index_map argument.

## Development Workflows

### Running Tests
```bash
# Install dependencies (includes pytest, pytest-cov)
pip install -e ".[dev]"

# Run tests with coverage (configured in pyproject.toml)
pytest

# Coverage threshold: 60% (fails below this)
```

Test files are in `tests/` and follow pattern `test_*.py`. Tests use pseudo-random data generators (`pseudo_data()`, `pseudo_weights()`).

### Project Configuration
- **Configuration**: `pyproject.toml` (single source of truth)
- **Versioning**: Uses `setuptools_scm` for automatic version from git tags
- Pytest config is in `[tool.pytest.ini_options]` section of `pyproject.toml`
- Dev dependencies in `[project.optional-dependencies]` section

### Building/Installing
```bash
# Editable install
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Code Conventions

### Type Annotations
Uses custom numpy type aliases defined in `matrix_handler.py`:
```python
Array1D_Float = Annotated[np.ndarray[ScalarType_co, np.float64], Literal['N']]
Array2D_Bool = Annotated[np.ndarray[ScalarType_co, bool], Literal['N', 'N']]
```

### Graph Structure Peculiarity
The graph implementation adds a **dummy target node** at position `dim` (size of matrix). This is why:
- `weights` array has length `N+1` (extra zero weight for target)
- `set_dummy_target()` expands matrix by 1
- Paths exclude target unless `trim=False` in `hdfs()`

### Result Storage Pattern
- Internally stores paths as `set` (order-independent)
- Returns sorted lists via `get_paths` property
- Uses `sort_index` field in `Result` dataclass for comparison operators

### Testing Pattern
Tests verify algorithm consistency by comparing HDFS vs WHDFS on same data with sorted weights (should yield identical results).

## Common Pitfalls

1. **Negative weights**: Both algorithms check for negative weights when `allow_subset=False` (cannot guarantee subset exclusion)
2. **Index confusion**: Results from sorted BAM are in sorted index space - must remap to get original indices
3. **Subset filtering**: `allow_subset=False` removes paths that are subsets of longer paths (default behavior)
4. **Source node**: Algorithms iterate through different source nodes (0 to N-1) to explore all paths

## Key Files for Reference
- **Tutorial**: `path_test.ipynb` - complete example workflow with visualization
- **Test patterns**: `tests/test_dfs.py` - shows typical usage with pseudo data
- **API surface**: `pathfinder/__init__.py` - exports `BinaryAcceptance`, `HDFS`, `WHDFS`, `Results`

## When Making Changes
- Update `pyproject.toml` for dependency changes
- Run tests before committing: `pytest`
- Version is automatically managed by `setuptools_scm` from git tags
- For algorithm changes, verify both HDFS and WHDFS produce expected results on sorted weights
- Visualisation changes affect `plot_results.py` - ensure matplotlib rcParams remain consistent
