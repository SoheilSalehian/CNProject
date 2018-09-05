package main

import (
	"fmt"
	"log"
	"math"
)

func sgdSolver(dataset DataSet, iterations int) []float64 {

	weights := make([]float64, len(dataset[0].X))

	for i := range weights {
		weights[i] = 0.0
	}

	input := make([][]float64, len(dataset))
	for i, data := range dataset {
		input[i] = data.X
	}

	output := make([]float64, len(dataset))
	for i, data := range dataset {
		output[i] = data.Y
	}

	for t := 1; t <= iterations; t++ {
		grads, preds, avgDeviation := gradients(dataset, weights)
		if t%1000 == 0 {
			fmt.Println("Iteration: ", t, "training RMSE: ", math.Abs(RMSE(preds, output)), "Average Deviation: ", avgDeviation)
			fmt.Println(weights)
		}

		if math.Abs(NRMSE(preds, output)) < 0.08 && *earlyStop == true {
			fmt.Printf("----Converged----")
			return weights
		}

		for j := range weights {
			weights[j] -= alpha * grads[j]
		}
	}
	return weights
}

func adamaxSolver(dataset DataSet, iterations int) []float64 {

	m := make([]float64, len(dataset))
	v := make([]float64, len(dataset))
	weights := make([]float64, len(dataset[0].X))

	for i := range weights {
		weights[i] = 0.0
	}

	input := make([][]float64, len(dataset))
	for i, data := range dataset {
		input[i] = data.X
	}

	output := make([]float64, len(dataset))
	for i, data := range dataset {
		output[i] = data.Y
	}

	for t := 1; t <= iterations; t++ {
		grads, preds, avgDeviation := gradients(dataset, weights)
		if t%1000 == 0 {
			fmt.Println("Iteration: ", t, "training RMSE: ", math.Abs(RMSE(preds, output)), "Average Deviation: ", avgDeviation)
			fmt.Println(weights)
		}

		if math.Abs(NRMSE(preds, output)) < 0.08 && *earlyStop == true {
			fmt.Printf("----Converged----")
			return weights
		}

		for j := range weights {
			lrt := alpha * (math.Sqrt(1.0 - math.Pow(beta2, float64(t)))) /
				(1.0 - math.Pow(beta1, float64(t)))

			m[j] = beta1*m[j] + (1.0-beta1)*grads[j]
			v[j] = math.Max(beta2*v[j], math.Abs(grads[j]))

			weights[j] -= lrt * (m[j] / (math.Sqrt(v[j]) + epsilon))
		}
	}
	return weights
}

//TODO: Add minibatches
func gradients(dataset DataSet, weights []float64) ([]float64, []float64, float64) {

	g := make([]float64, len(weights))

	// FIXME: too many passes on dataset
	input := make([][]float64, len(dataset))
	for i, data := range dataset {
		input[i] = data.X
	}

	preds, err := prediction(input, weights)
	if err != nil {
		log.Fatal(err)
	}

	errSum := 0.0

	errs := make([]float64, len(preds))
	for j, data := range dataset {
		errs[j] = data.Y - preds[j]
		errSum += errs[j] / data.Y
	}

	g, err = dot(transpose(input), errs)
	if err != nil {
		fmt.Println("dot product for gradient calculations failed.")
		log.Fatal(err)
	}

	for k := range g {
		g[k] /= float64(len(dataset)) * -1.0
	}

	return g, preds, errSum / float64(len(dataset))
}

func prediction(x [][]float64, weights []float64) ([]float64, error) {

	preds, err := dot(x, weights)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	return preds, nil
}
