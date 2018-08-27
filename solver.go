package main

import (
	"fmt"
	"log"
	"math"
)

func adamSolver(dataset DataSet) []float64 {

	m := make([]float64, len(dataset))
	v := make([]float64, len(dataset))
	weights := make([]float64, len(dataset[0].X))

	for i := range weights {
		weights[i] = 4.0
	}

	input := make([][]float64, len(dataset))
	for i, data := range dataset {
		input[i] = data.X
	}

	output := make([]float64, len(dataset))
	for i, data := range dataset {
		output[i] = data.Y
	}

	t := 1

	e := 0.0
	for {
		grads, preds := gradients(dataset, weights, len(dataset))
		if t%1000 == 0 {
			fmt.Println("Iteration: ", t, "trainig NRMSE: ", math.Abs(NRMSE(preds, output)))
		}

		e += math.Abs(NRMSE(preds, output))

		if math.Abs(NRMSE(preds, output)) < 0.08 {
			fmt.Printf("----Converged----")
			return weights
		}

		for j := range weights {
			lrt := alpha * (math.Sqrt(1.0 - math.Pow(beta2, float64(t)))) /
				(1.0 - math.Pow(beta1, float64(t)))

			m[j] = beta1*m[j] + (1.0-beta1)*grads[j]
			v[j] = beta2*v[j] + (1.0-beta2)*math.Pow(grads[j], 2.0)

			weights[j] -= lrt * (m[j] / (math.Sqrt(v[j]) + epsilon))
		}
		t += 1
	}
	return weights
}

//TODO: Add minibatches
func gradients(dataset DataSet, weights []float64, batchSize int) ([]float64, []float64) {

	g := make([]float64, len(weights))

	// FIXME: too many passes on dataset
	input := make([][]float64, len(dataset))
	for i, data := range dataset[0:batchSize] {
		input[i] = data.X
	}

	preds, err := prediction(input, weights)
	if err != nil {
		log.Fatal(err)
	}

	errs := make([]float64, len(preds))
	for j, data := range dataset[0:batchSize] {
		errs[j] = data.Y - preds[j]
	}

	g, err = dot(transpose(input), errs)
	if err != nil {
		fmt.Println("dot product for gradient calculations failed.")
		log.Fatal(err)
	}

	for k := range g {
		g[k] /= float64(len(dataset)) * -1.0
	}

	return g, preds
}

func prediction(x [][]float64, weights []float64) ([]float64, error) {

	preds, err := dot(x, weights)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	return preds, nil
}
