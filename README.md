# PathFinder

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-83%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-93.56%25-brightgreen.svg)](pyproject.toml)

**Finding optimal combinations of features with pairwise constraints**

PathFinder is a Python library for identifying optimal subsets of elements based on pairwise relations and weights. Originally developed for combining particle physics analyses at ATLAS and CMS, it solves the general problem of feature selection where pairwise correlations constrain which features can be combined.

## Overview

The core challenge PathFinder addresses is combinatorial optimisation under constraints: given $n$ features with pairwise compatibility constraints, find the optimal subset that maximises a cumulative metric (e.g., statistical significance, information content) whilst respecting those constraints.

For $n$ features, there are $2^n - (n+1)$ possible subsets to evaluate. PathFinder reduces this exponential complexity to near $O(n \log n)$ in typical cases by:

1. **Binary Acceptance Matrix (BAM)**: Encoding pairwise constraints as a boolean matrix
2. **Hereditary Depth-First Search (HDFS)**: Efficiently exploring only valid combinations
3. **Weighted HDFS (WHDFS)**: Optimising for weighted objectives with early termination

### Key Features

- **Efficient combinatorial search**: Avoids generating invalid combinations
- **Flexible constraint specification**: Correlation matrices, mutual information, or custom metrics
- **Weight-based optimisation**: Find subsets maximising additive metrics
- **Subset filtering**: Optionally exclude subsets of larger valid combinations
- **Index remapping**: Seamless handling of sorted/unsorted feature orderings
- **Visualisation tools**: Matplotlib-based plotting of acceptance matrices and result paths

## Installation

### From PyPI (when published)

```bash
pip install pathfinder
```

### From source

```bash
git clone https://github.com/J-Yellen/PathFinder.git
cd PathFinder
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
pytest  # Run tests
```

## Quick Start

```python
import numpy as np
import pathfinder as pf

# Define pairwise relations (e.g., correlation matrix)
n_features = 20
correlation_matrix = np.random.rand(n_features, n_features)
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric

# Define feature weights (e.g., statistical significance)
weights = np.random.rand(n_features)

# Create Binary Acceptance Matrix with threshold
bam = pf.BinaryAcceptance(correlation_matrix, weights=weights, threshold=0.5)

# Sort by weights for optimal performance
index_map = bam.sort_bam_by_weight()

# Find optimal combinations
whdfs = pf.WHDFS(bam, top=5, allow_subset=False)
whdfs.find_paths(verbose=True)

# Remap results to original indices (optional - get_paths does this automatically)
results = whdfs.remap_path()  # index_map is stored in bam

# Access results
print(f"Best combination: {results.get_paths[0]}")
print(f"Combined weight: {results.get_weights[0]}")
```

## Theoretical Background

PathFinder implements algorithms described in Chapter 6 of [Yellen (2025)](https://doi.org/10.5525/gla.thesis.85009), originally developed for combining particle physics searches for Beyond Standard Model physics.

### The Hereditary Condition

The key insight is that valid feature combinations can be viewed as paths through a directed acyclic graph (DAG) where:

- **Nodes** represent features
- **Edges** connect features that can be combined (correlation/overlap below threshold)
- **Paths** represent valid feature subsets

The *hereditary condition* requires that each new feature added to a subset must be compatible with *all* previously selected features. This is enforced by maintaining an intersection of allowed features at each step:

$$S_c = A_c \cap S_{c-1}$$

where $S_c$ is the set of compatible features at step $c$, and $A_c$ are features compatible with the current feature.

### Binary Acceptance Matrix (BAM)

Given a pairwise relation matrix $\rho$ (e.g., correlation) and threshold $T$, the BAM $B$ is defined as:

$$
\begin{aligned}
B_{ij} &= 1 \quad \text{if } |\rho_{ij}| < T \text{ (features can be combined)} \\
B_{ij} &= 0 \quad \text{otherwise (features are too correlated)}
\end{aligned}
$$

The fraction of allowed combinations is:

$$f_{A} = \frac{2}{n \cdot (n-1)} \sum_{i < j} B_{ij}$$

### Algorithms

#### HDFS (Hereditary Depth-First Search)

Finds *all* valid combinations by:
1. Starting from each feature as a source
2. Recursively building paths using only compatible features
3. Terminating at a universal "sink" node

**Complexity**: Depends on $f_A$ (allowed fraction). For sparse matrices (low $f_A$), near-linear scaling.

**Use case**: When you need to know all possible combinations.

#### WHDFS (Weighted Hereditary Depth-First Search)

Optimises for weighted objectives by:
1. Sorting features by weight (most important first)
2. Maintaining best-path-so-far
3. **Early termination**: Abandoning branches where:

$$\text{current weight} + \text{max remaining weight} < \text{best weight}$$

**Complexity**: Typically $O(n \log n)$ for practical $f_A$ values. Up to 1000× faster than HDFS for $f_A = 0.75$.

**Use case**: When you want the top-k best combinations.

### Ordering Optimisation

Sorting the BAM by decreasing feature weights ensures:
- Optimal paths are likely found early
- Early termination triggers sooner
- Typical use only requires searching from first ~10 features

The `sort_bam_by_weight()` method returns an index map to convert results back to original ordering.

## Usage Examples

### Example 1: Feature Selection for Machine Learning (Simple)

```python
import pathfinder as pf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import f_classif

# Load and prepare data
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
correlation = np.corrcoef(X.T)

# Define feature weights
f_stats, _ = f_classif(X, data.target)
weights = f_stats / f_stats.sum()

# Find best feature combinations (one function call!)
results = pf.find_best_combinations(
    matrix=correlation,
    weights=weights,
    threshold=0.7,
    top=10
)

# Print results
print("Top 10 feature combinations:")
for i, (path, weight) in enumerate(zip(results.get_paths, results.get_weights), 1):
    feature_names = [data.feature_names[j] for j in path]
    print(f"{i}. {feature_names} (weight: {weight:.3f})")
```

### Example 1b: Feature Selection (Advanced Control)

```python
import pathfinder as pf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import f_classif

# Load and prepare data
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
correlation = np.corrcoef(X.T)

# Define feature weights
f_stats, _ = f_classif(X, data.target)
weights = f_stats / f_stats.sum()

# Create BAM and search (automatic sorting and remapping)
bam = pf.BinaryAcceptance(correlation, weights=weights, threshold=0.7)
whdfs = pf.WHDFS(bam, top=10, allow_subset=False)  # auto_sort=True by default
whdfs.find_paths()

# Results are automatically in original index space
print("Top 10 feature combinations:")
for i, (path, weight) in enumerate(zip(whdfs.get_paths, whdfs.get_weights), 1):
    feature_names = [data.feature_names[j] for j in path]
    print(f"{i}. {feature_names} (weight: {weight:.3f})")
```

### Example 2: Particle Physics Signal Region Combination

```python
import pathfinder as pf
import numpy as np

# Signal region overlaps (from Monte Carlo simulation)
n_regions = 50
overlap_matrix = np.random.rand(n_regions, n_regions)
overlap_matrix = (overlap_matrix + overlap_matrix.T) / 2

# Expected signal significance per region
expected_significance = np.random.exponential(2.0, n_regions)

# Find non-overlapping regions maximising combined significance
results = pf.find_best_combinations(
    matrix=overlap_matrix,
    weights=expected_significance,
    threshold=0.1,  # Max 10% overlap
    top=5,
    runs=10  # Only search from top 10 regions
)

print(f"Optimal signal regions: {results.get_paths[0]}")
print(f"Combined significance: {results.get_weights[0]:.2f}σ")
```

### Example 3: Visualisation

```python
import pathfinder as pf
from pathfinder import plot_results
import matplotlib.pyplot as plt

# Create and solve problem
bam = pf.BinaryAcceptance(correlation_matrix, weights=weights, threshold=0.5)
whdfs = pf.WHDFS(bam, top=5)  # Automatic sorting for optimal performance
whdfs.find_paths()

# Plot BAM with overlaid results (note: plot shows sorted indices)
fig, ax = plot_results.plot(bam, whdfs, size=12)
plt.savefig('pathfinder_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Access results in original index space
print(f"Best path (original indices): {whdfs.get_paths[0]}")
```

## API Reference

### High-Level Functions

#### `find_best_combinations`

Convenience function for the complete PathFinder workflow.

```python
find_best_combinations(matrix, weights=None, threshold=None, top=10,
                      allow_subset=False, runs=None, labels=None,
                      algorithm='whdfs', verbose=False)
```

**Parameters:**
- `matrix`: Square 2D array of pairwise relations (boolean or float)
- `weights`: Optional array of feature weights (default: uniform)
- `threshold`: Required for float matrices - values below threshold are compatible
- `top`: Number of top results to return (default: 10)
- `allow_subset`: Allow paths that are subsets of longer paths (default: False)
- `runs`: Number of source nodes to search from (default: all)
- `labels`: Optional feature labels
- `algorithm`: 'whdfs' (default, faster) or 'hdfs' (exhaustive)
- `verbose`: Print results to console (default: False)

**Returns:**
- `Results`: Results object with automatically remapped paths

**Note:** This function automatically handles sorting and remapping, returning results in the original index space.

### Core Classes

#### `BinaryAcceptance`

Converts pairwise relation matrices to graph representation.

```python
BinaryAcceptance(matrix, weights=None, labels=None,
                 threshold=None, allow_negative_weights=False)
```

**Parameters:**
- `matrix`: Square 2D array of boolean or float values
- `weights`: Optional array of feature weights (default: uniform)
- `labels`: Optional feature labels
- `threshold`: Required for float matrices (values < threshold are compatible)
- `allow_negative_weights`: Allow negative weights (requires `allow_subset=True`)

**Key Methods:**
- `sort_bam_by_weight()`: Sort by descending weights, returns index map
- `get_weight(path)`: Calculate total weight for path
- `reset_source(source)`: Change source node

#### `HDFS` (Hereditary Depth-First Search)

Finds all valid combinations.

```python
HDFS(binary_acceptance_obj, top=10, allow_subset=False)
```

**Parameters:**
- `binary_acceptance_obj`: BinaryAcceptance instance
- `top`: Number of top results to retain
- `allow_subset`: If False, exclude paths that are subsets of longer paths

**Key Methods:**
- `find_paths(runs=None, source_node=0, verbose=False)`: Execute search
  - `runs`: Number of source nodes to search from (default: all)
  - `source_node`: Starting source node index
  - `verbose`: Print results

#### `WHDFS` (Weighted Hereditary Depth-First Search)

Optimised search for weighted objectives with automatic sorting.

```python
WHDFS(binary_acceptance_obj, top=10, allow_subset=False, auto_sort=True)
```

**Parameters:**
- `binary_acceptance_obj`: BinaryAcceptance instance
- `top`: Number of top results to retain
- `allow_subset`: If False, exclude paths that are subsets of longer paths
- `auto_sort`: Automatically sort by weights for optimal performance (default: True)

**Key Features:**
- Automatic weight-based sorting when `auto_sort=True` (default)
- Results accessed via `get_paths` and `get_weights` are automatically remapped to original indices
- Up to 1000× faster than HDFS for typical problems
- Emits performance warning if `auto_sort=False` with non-uniform weights

**Key Methods:**
- `find_paths(runs=None, source_node=0, verbose=False)`: Execute search
  - `runs`: Number of source nodes to search from (default: all)
  - `source_node`: Starting source node index
  - `verbose`: Print results
- `get_sorted_paths()`: Get paths in sorted index space (for debugging)
- `get_sorted_results()`: Get complete Results object in sorted index space

**Important Note on Index Spaces:**
When using `WHDFS` with `auto_sort=True` (default):
- Internally, the algorithm works in **sorted index space** (weight-ordered)
- `get_paths` and `get_weights` automatically return results in **original index space**
- `get_sorted_paths()` and `get_sorted_results()` return **sorted index space** (for advanced use)
- The `__str__()` representation displays paths in original index space for user-friendliness

**Plotting Behavior:**
When using `plot_results.plot()` with WHDFS:
- Default (`plot_sorted=False`): Displays matrix in original order with paths in original indices
  - Best for visual comparison with input data
- `plot_sorted=True`: Displays matrix in sorted order with paths in sorted indices
  - Best for understanding algorithm internals and performance
- The function automatically handles index space alignment

#### `Results`

Container for path/weight pairs with sorting and filtering.

**Properties:**
- `get_paths`: List of paths (sorted lists of indices)
  - For `WHDFS` with `auto_sort=True`: automatically remapped to original indices
  - For `HDFS` or manual sorting: requires `remap_path()` call
- `get_weights`: Corresponding weights
- `best`: Highest-weighted Result
- `top`: Number of results to retain (settable)

**Methods:**
- `remap_path(index_map=None, weight_offset=0.0)`: Convert to original indices (index_map auto-detected from bam)
  - Not needed for `WHDFS` with `auto_sort=True` (automatic)
  - Required for `HDFS` or when using manual `sort_bam_by_weight()`
- `to_dict()` / `from_dict()`: Serialisation
- `from_list_of_results(result_list)`: Create from list of Result objects
- `from_list_of_dicts(result_list)`: Create from list of dictionaries
- `to_json()` / `from_json()`: JSON I/O

**Equality Comparison:**
The `==` operator compares Results objects with proper handling of:
- Index space remapping (automatically uses `get_paths` for WHDFS)
- Floating-point tolerance for weights (uses `np.allclose` with rtol=1e-9)
- Path set comparison (order-independent)

## Configuration & Best Practices

### Handling Negative Weights

Negative weights can represent evidence against a hypothesis. Two options:

**Option 1: Rescale to positive** (required if `allow_subset=False`)
```python
constant_shift = abs(min(weights)) + 1
weights_positive = weights + constant_shift
bam = pf.BinaryAcceptance(matrix, weights=weights_positive)
# ... run algorithm ...
results = hdfs.remap_path(weight_offset=constant_shift)  # index_map optional
```

**Option 2: Allow negative weights** (requires `allow_subset=True`)
```python
bam = pf.BinaryAcceptance(matrix, weights=weights,
                           allow_negative_weights=True)
hdfs = pf.HDFS(bam, allow_subset=True)
```

### Subset Filtering

By default (`allow_subset=False`), PathFinder excludes paths that are proper subsets of other valid paths. This is crucial for preventing statistical biases:

- **Cherry-picking prevention**: By excluding subsets, you cannot selectively report only the favorable combinations whilst omitting features that worsen the result. You must include all compatible features.
- **Look-elsewhere effect mitigation**: Searching through many possible subsets and selecting the best-looking one inflates false positive rates. Requiring maximal sets reduces the effective number of independent tests.
- **Honest reporting**: Forces inclusion of all compatible features, even those with poor individual performance, preventing selection bias.

**Important**: When `allow_subset=False`, negative weights are not permitted because the subset-exclusion logic relies on additive weights being monotonic (adding more features always increases total weight). This is why you must rescale negative weights to positive values using:

```python
constant_shift = abs(min(weights)) + 1
weights_positive = weights + constant_shift
```

Enable `allow_subset=True` when:
- You have negative weights and want to allow features that decrease the metric
- Data reduction is the primary goal (finding the most informative subset, not necessarily the largest)
- You're willing to accept the increased risk of selection bias

### Performance Tips

1. **Use `WHDFS` with default settings** (automatic sorting enabled):
   ```python
   whdfs = pf.WHDFS(bam, top=10)  # auto_sort=True by default
   ```
   This provides up to 1000× speedup over HDFS and handles index remapping automatically.

2. **Or use the convenience function** for simplest workflow:
   ```python
   results = pf.find_best_combinations(matrix, weights, threshold=0.7, top=10)
   ```

3. **Limit source nodes** for large problems:
   ```python
   whdfs.find_paths(runs=10)  # Only search from top 10 features
   ```
   Typically, optimal paths are found in the first 10 high-weight nodes.

4. **Choose appropriate threshold**: Lower threshold (stricter compatibility) → faster search

5. **Use WHDFS over HDFS** when you only need top results (not exhaustive enumeration)

6. **Advanced: Manual control** (if you need to disable auto-sort):
   ```python
   bam = pf.BinaryAcceptance(matrix, weights=weights)
   index_map = bam.sort_bam_by_weight()
   whdfs = pf.WHDFS(bam, top=10, auto_sort=False)
   whdfs.find_paths()
   results = whdfs.remap_path()  # No need to pass index_map
   ```

## Testing

PathFinder uses pytest with 93.56% code coverage and 83 passing tests:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/pathfinder --cov-report=html

# Run specific test file
pytest tests/test_dfs.py -v

# Run edge case tests
pytest tests/test_edge_cases.py -v
```

**Test Suite Coverage:**
- `test_dfs.py`: HDFS/WHDFS algorithm validation
- `test_matrix_handler.py`: BinaryAcceptance functionality
- `test_result.py`: Results class operations
- `test_plot_results.py`: Visualization correctness (including paths-on-valid-edges validation)
- `test_edge_cases.py`: Boundary conditions (fully connected/disconnected, minimal graphs, thresholds)

**Critical Validation:**
The test suite includes `test_paths_dont_land_on_black_squares()` which validates that paths never traverse invalid edges, preventing the critical plotting bug from recurring.

## Citation

If you use PathFinder in your research, please cite:

```bibtex
@phdthesis{Yellen2025,
  author  = {Yellen, Jamie},
  title   = {Search strategies for Beyond Standard Model physics at the LHC},
  school  = {University of Glasgow},
  year    = {2025},
  doi     = {10.5525/gla.thesis.85009}
}
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Ensure tests pass (`pytest`)
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/J-Yellen/PathFinder.git
cd PathFinder
pip install -e ".[dev]"
pytest
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgements

Developed as part of PhD research at the University of Glasgow SUPA School of Physics and Astronomy, supported by the ATLAS and CMS collaborations at CERN.

## Links

- **Repository**: https://github.com/J-Yellen/PathFinder
- **Documentation**: https://github.com/J-Yellen/PathFinder/wiki
- **Issues**: https://github.com/J-Yellen/PathFinder/issues
- **PhD Thesis**: https://doi.org/10.5525/gla.thesis.85009

