#!/bin/bash

for SCRIPT in $(ls NeuralODE/run); do sbatch --time=29-00:00:00 NeuralODE/run/$SCRIPT; done
