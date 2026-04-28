[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xWOTuKMG)
# Gradient Descent and Nesterov Methods

An advanced C++ optimization framework implementing gradient descent and Nesterov methods with configurable learning rate schedules and gradient computation strategies.

## Overview

This project provides a flexible optimization toolkit that supports:
- **Two optimization methods**: Standard Gradient Descent and Nesterov Method
- **Three learning rate strategies**: Exponential Decay, Inverse Decay, and Line Search (Armijo's rule)
- **Two gradient computation modes**: Analytic (from config) and Finite Difference approximation
- **Dynamic configuration**: Parameters can be specified via configuration file at runtime

## Files

### main.cpp
Complete C++ implementation featuring:

**Core Components:**
- `norm()`, `norm_squared()` - Vector norm computations (L2 norm)
- `operator-()`, `operator+()`, `operator*()` - Vector algebra operations
- `Gradient<AlphaRule, GradRule>()` - Template-based gradient descent algorithm
- `Nesterov<AlphaRule, GradRule>()` - Template-based Nesterov algorithm
- `finite_diff_grad()` - Numerical gradient approximation (central difference, h=1e-6)
- `line_search()` - Armijo's rule for adaptive step size selection
- `exp_decay()`, `inv_decay()` - Learning rate scheduling functions

**Configuration Management:**
- `read_config()` - Parses configuration file using muparserx expression parser
- `parseAlphaRule()`, `parseGradRule()` - String-to-enum conversion functions
- `runGradient()`, `runNesterov()` - Dispatchers handling all template specializations

**Data Structures:**
- `Params` struct - Encapsulates all optimization parameters
- `AlphaRule` enum - Learning rate update strategies (EXP_DECAY, INV_DECAY, LINE_SEARCH)
- `GradRule` enum - Gradient computation methods (ANALYTIC, FINITE_DIFF)

### config.txt
Configuration file specifying optimization parameters. Format (10 lines, comments start with `#`):

```
# Line 1: Objective function expression (muparserx syntax, e.g., x1^2 + x2^2)
x1*x2 + 4*x1^4 + x2^2 + 3*x1

# Line 2: Gradient expressions - comma-separated for each component (e.g., 2*x1, 2*x2)
x2 + 16*x1^3, x1 + 2*x2

# Line 3: Initial point - space-separated values (e.g., 1.0 1.0)
1.0 1.0

# Line 4: Step size tolerance (convergence criterion)
1e-6

# Line 5: Residual tolerance - gradient norm threshold (convergence criterion)
1e-6

# Line 6: Initial learning rate
0.1

# Line 7: Maximum iterations
1000

# Line 8: Sigma for Armijo line search (optional, default 0.5)
0.1

# Line 9: Alpha rule - EXP_DECAY, INV_DECAY, or LINE_SEARCH (optional, default LINE_SEARCH)
LINE_SEARCH

# Line 10: Gradient rule - ANALYTIC or FINITE_DIFF (optional, default FINITE_DIFF)
FINITE_DIFF
```

**Variable Notation:**
- `x1` and `x2` represent the two optimization variables
- muparserx syntax: use `^` for exponentiation (e.g., `x1^2`), `*` for multiplication, `+` for addition

## Build Instructions

### Required Dependencies
- **muparserx**: Mathematical expression parser library
  ```bash
  sudo apt install libmuparserx-dev libmuparserx4.0.11
  ```
- **g++**: C++23 compiler with optimization

### Clone and Build
```bash
git clone https://github.com/YOUR-USERNAME/challenge1-ZiadKandil.git
cd challenge1-ZiadKandil
make
```

### Clean
```bash
make clean          # Remove only project binaries
```

## Usage

### With Configuration File
```bash
./main config.txt
```
Loads optimization parameters from `config.txt` and runs both Gradient and Nesterov methods.

### Without Configuration File (Hardcoded Parameters)
```bash
./main
```
Uses default hardcoded parameters for demonstration.

### Example Output
```
Configuration loaded successfully from config.txt
Running Gradient method with Line Search learning rate and Finite Difference gradient.
Step size convergence achieved using gradient method at iteration 106
Minimum value using Gradient method is: -1.37233
Running Nesterov method with Line Search learning rate and Finite Difference gradient.
Residual convergence achieved using Nesterov method at iteration 87
Minimum value using Nesterov method is: -1.37233
```

## Algorithm Details

### Gradient Descent
Iteratively updates: **x := x - α * ∇f(x)**

Convergence criteria:
- Residual: ||∇f(x)|| < res_tol
- Step size: ||x - x_old|| < step_tol

### Nesterov Method
Uses momentum with Nesterov's acceleration:
- **y := x - α * ∇f(x)**
- **x := y + η * (y - y_old)** with adaptive momentum coefficient η

Provides faster convergence than standard gradient descent.

### Learning Rate Strategies
- **EXP_DECAY**: α(k) = α₀ * exp(-decay_rate * k)
- **INV_DECAY**: α(k) = α₀ / (1 + decay_rate * k)
- **LINE_SEARCH**: Adaptive α via Armijo's rule with backtracking

## Documentation

Comprehensive Doxygen comments are included for all functions, structs, and enums. To generate HTML documentation:

```bash
doxygen -g Doxyfile
doxygen Doxyfile
```

## Configuration Examples

### Example 1: Simple Quadratic Function
```
# Function: f(x) = x1^2 + x2^2
x1^2 + x2^2
# Gradient: ∇f = (2*x1, 2*x2)
2*x1, 2*x2
# Initial point
0.5 0.5
# Tolerances
1e-6
1e-6
# Learning rate and iterations
0.1
1000
```

### Example 2: With Different Learning Rate Schedule
```
# Function
x1^2 + 2*x1*x2 + x2^2
# Gradient
2*x1 + 2*x2, 2*x1 + 2*x2
# Initial point
1.0 1.0
# Tolerances
1e-8
1e-8
# Parameters
0.05
5000
0.1
# Learning rate rule: exponential decay
EXP_DECAY
# Gradient rule: analytic
ANALYTIC
```

## Project Structure
```
challenge1-ZiadKandil/
├── main.cpp          # Main implementation with algorithms
├── config.txt        # Configuration file
├── Makefile          # Build configuration
└── README.md         # This file
```

## Performance Notes

- **Nesterov acceleration**: Typically 10-30% faster convergence than standard gradient descent
- **Line Search**: More robust but computationally expensive (requires extra function evaluations per iteration)
- **Exponential/Inverse Decay**: Faster but may require parameter tuning for different problems
- **Finite Difference Gradient**: More general but requires accurate step size selection (default h=1e-6)
