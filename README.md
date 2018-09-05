# CNProject

The implementation of multivariate linear regression using the Adamax solver.

## How to use

The CLI tool can be used by invoking:

```
Usage of ./CNProject:
  -cpuProfile
    	To enable CPU profiling.
  -earlyStop
    	Enable/Disable early stopping based on NRMSE criteria. (default true)
  -iters int
    	Number of iterations to run the regression (default 40000)
  -solver string
    	type of solver used: 'sdg' or 'adamax' supported currently. (default "adamax")
```

## Dataset

The [dataset](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware) is imported via csv.
