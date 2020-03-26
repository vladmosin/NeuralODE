#!/bin/bash

for SCRIPT in $"ls run"; do sbatch $SCRIPT; done