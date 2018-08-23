package main

import (
	"fmt"
	"log"
	"math"
)

type DataSet []Data

type Data struct {
	X []float64
	Y float64
}

type Regression struct {
	weights    []float64
	iterations int
}

const (
	alpha   float64 = 0.001
	beta1   float64 = 0.9
	beta2   float64 = 0.999
	epsilon float64 = 1e-8
)

func (r *Regression) Predict(x [][]float64) []float64 {
	return prediction(x, r.weights)
}

func (r *Regression) Fit(dataset DataSet) error {
	if r.iterations == 0 {
		log.Fatal("Number of iterationss has not been defined.")
	}

	r.weights = adamSolver(dataset, r.iterations)
	return nil
}

func adamSolver(dataset DataSet, iterations int) []float64 {

	m := make([]float64, len(dataset))
	v := make([]float64, len(dataset))
	weights := make([]float64, len(dataset))

	for i := range dataset {
		weights[i] = 1.0
	}

	for t := 1; t < iterations; t++ {
		grad := gradient(dataset, weights, len(dataset))
		for j := range grad {
			lrt := alpha * (math.Sqrt(1.0 - math.Pow(beta2, float64(t)))) /
				(1.0 - math.Pow(beta1, float64(t)))
			m[j] = beta1*m[j] + (1.0-beta1)*grad[j]
			v[j] = beta2*v[j] + (1.0-beta2)*math.Pow(grad[j], 2.0)

			weights[j] -= lrt * (m[j] / (math.Sqrt(v[j]) + epsilon))
		}
	}
	return weights
}

func gradient(dataset DataSet, weights []float64, batchSize int) []float64 {

	g := make([]float64, len(weights))
	for i := range weights {
		g[i] = 0.0
	}

	// FIXME: too many passes on dataset
	input := make([][]float64, len(dataset))
	for i, data := range dataset[0:batchSize] {
		input[i] = data.X
	}

	preds := prediction(input, weights)
	for j, data := range dataset[0:batchSize] {
		g[j] += ((preds[j] - data.Y) * data.X[j])
	}

	return g
}

func prediction(x [][]float64, weights []float64) []float64 {
	preds := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[0]); j++ {
			preds[i] += x[i][j] * weights[i]
		}
	}

	return preds
}

func meanSquaredError(y, predY []float64) float64 {
	var total float64
	for i := 0; i < len(y); i++ {
		total += (y[i] - predY[i]) * (y[i] - predY[i])
	}

	return total / float64(len(y))
}

func main() {
	xTrain := [][]float64{[]float64{1, 1, 1, 1, 1}, []float64{2, 2, 2, 2, 2}, []float64{3, 3, 3, 3, 3}, []float64{4, 4, 4, 4, 4}}
	xTest := [][]float64{[]float64{5, 5, 5, 5, 5}, []float64{6, 6, 6, 6, 6}, []float64{7, 7, 7, 7, 7}, []float64{8, 8, 8, 8, 8}}
	yTest := []float64{5, 6, 7, 8}

	dsTrain := make(DataSet, len(xTrain))
	for i, d := range xTrain {
		dsTrain[i].X = d
		dsTrain[i].Y = d[0]
	}

	dsTest := make(DataSet, len(xTest))
	for i, d := range xTest {
		dsTest[i].X = d
		dsTest[i].Y = d[0]
	}

	lr := Regression{iterations: 4000}
	lr.Fit(dsTrain)

	pred := lr.Predict(xTest)

	fmt.Println(pred)
	fmt.Println("MSE: ", meanSquaredError(yTest, pred))
}
