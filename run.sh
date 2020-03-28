#!/bin/bash

for SCRIPT in $(ls NeuralODE/run); do sbatch NeuralODE/run/$SCRIPT; done
