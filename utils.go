package main

import (
	"errors"
	"fmt"
	"math"
)

func RMSE(y, predY []float64) float64 {
	var total float64
	for i := 0; i < len(y); i++ {
		total += math.Pow((y[i] - predY[i]), 2)
	}

	return math.Sqrt(total / float64(len(y)))
}

func NRMSE(y, predY []float64) float64 {
	var total float64
	for i := 0; i < len(y); i++ {
		total += math.Pow((y[i] - predY[i]), 2)
	}
	min, max := minimum(y), maximum(y)

	return (math.Sqrt(total / float64(len(y)))) / (max - min)
}

func transpose(m [][]float64) [][]float64 {
	r := make([][]float64, len(m[0]))
	for x, _ := range r {
		r[x] = make([]float64, len(m))
	}
	for y, s := range m {
		for x, e := range s {
			r[x][y] = e
		}
	}
	return r
}

func minimum(xx []float64) float64 {
	// fmt.Println(xx)
	min := xx[0]
	for _, x := range xx {
		if x < min {
			min = x
		}
	}
	return min
}

func maximum(xx []float64) float64 {
	max := xx[0]
	for _, x := range xx {
		if x > max {
			max = x
		}
	}
	return max
}

func dot(x [][]float64, y []float64) ([]float64, error) {
	if len(x[0]) != len(y) {
		fmt.Println("X: ", len(x[0]), "Y: ", len(y))
		return nil, errors.New("Incorrect dimensions to do the dot product.")
	}

	out := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(y); j++ {
			if len(out) < 1 {
				out = make([]float64, len(y))
			}
			out[i] += x[i][j] * y[j]
		}
	}
	return out, nil
}
