#!/bin/sh


# ============================================================================
#	MNIST
# ============================================================================

# # Standard Neural Net
sbatch -p besteffort run.sh NN --data mnist --batch-size 128


# # Tangent Propagation vs trade-off
sbatch -p besteffort run.sh TP --data mnist --batch-size 128 --trade-off 0.0
sbatch -p besteffort run.sh TP --data mnist --batch-size 128 --trade-off 0.001
sbatch -p besteffort run.sh TP --data mnist --batch-size 128 --trade-off 0.01
sbatch -p besteffort run.sh TP --data mnist --batch-size 128 --trade-off 0.1
sbatch -p besteffort run.sh TP --data mnist --batch-size 128 --trade-off 1.0
sbatch -p besteffort run.sh TP --data mnist --batch-size 128 --trade-off 10.0
sbatch -p besteffort run.sh TP --data mnist --batch-size 128 --trade-off 100.0


# # Data augmentation vs width
sbatch -p besteffort run.sh ANN --data mnist --batch-size 128 --width 5 --n-augment 2
sbatch -p besteffort run.sh ANN --data mnist --batch-size 128 --width 5 --n-augment 5
sbatch -p besteffort run.sh ANN --data mnist --batch-size 128 --width 10 --n-augment 2
sbatch -p besteffort run.sh ANN --data mnist --batch-size 128 --width 10 --n-augment 5
sbatch -p besteffort run.sh ANN --data mnist --batch-size 128 --width 15 --n-augment 2
sbatch -p besteffort run.sh ANN --data mnist --batch-size 128 --width 15 --n-augment 5


# # Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 0.0
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 0.001
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 0.01
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 100.0


# # Pivot Adversarial Network vs width
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data mnist --batch-size 128 --width 15 --trade-off 1.0

# ============================================================================
#	FASHION MNIST
# ============================================================================

# Standard Neural Net
sbatch -p besteffort run.sh NN --data fashion-mnist --batch-size 128


# Tangent Propagation vs trade-off
sbatch -p besteffort run.sh TP --data fashion-mnist --batch-size 128 --trade-off 0.0
sbatch -p besteffort run.sh TP --data fashion-mnist --batch-size 128 --trade-off 0.001
sbatch -p besteffort run.sh TP --data fashion-mnist --batch-size 128 --trade-off 0.01
sbatch -p besteffort run.sh TP --data fashion-mnist --batch-size 128 --trade-off 0.1
sbatch -p besteffort run.sh TP --data fashion-mnist --batch-size 128 --trade-off 1.0
sbatch -p besteffort run.sh TP --data fashion-mnist --batch-size 128 --trade-off 10.0
sbatch -p besteffort run.sh TP --data fashion-mnist --batch-size 128 --trade-off 100.0


# Data augmentation vs width
sbatch -p besteffort run.sh ANN --data fashion-mnist --batch-size 128 --width 5 --n-augment 2
sbatch -p besteffort run.sh ANN --data fashion-mnist --batch-size 128 --width 5 --n-augment 5
sbatch -p besteffort run.sh ANN --data fashion-mnist --batch-size 128 --width 10 --n-augment 2
sbatch -p besteffort run.sh ANN --data fashion-mnist --batch-size 128 --width 10 --n-augment 5
sbatch -p besteffort run.sh ANN --data fashion-mnist --batch-size 128 --width 15 --n-augment 2
sbatch -p besteffort run.sh ANN --data fashion-mnist --batch-size 128 --width 15 --n-augment 5


# Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.0
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.001
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.01
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 100.0


# Pivot Adversarial Network vs width
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data fashion-mnist --batch-size 128 --width 15 --trade-off 1.0


# ============================================================================
#	HIGGS UCI
# ============================================================================

# # Standard Neural Net
sbatch -p besteffort run.sh NN --data higgs-uci --batch-size 1024


# # Tangent Propagation vs trade-off
sbatch -p besteffort run.sh TP --data higgs-uci --batch-size 1024 --trade-off 0.0
sbatch -p besteffort run.sh TP --data higgs-uci --batch-size 1024 --trade-off 0.001
sbatch -p besteffort run.sh TP --data higgs-uci --batch-size 1024 --trade-off 0.01
sbatch -p besteffort run.sh TP --data higgs-uci --batch-size 1024 --trade-off 0.1
sbatch -p besteffort run.sh TP --data higgs-uci --batch-size 1024 --trade-off 1.0
sbatch -p besteffort run.sh TP --data higgs-uci --batch-size 1024 --trade-off 10.0
sbatch -p besteffort run.sh TP --data higgs-uci --batch-size 1024 --trade-off 100.0


# # Data augmentation vs width
sbatch -p besteffort run.sh ANN --data higgs-uci --batch-size 1024 --width 0.01 --n-augment 2
sbatch -p besteffort run.sh ANN --data higgs-uci --batch-size 1024 --width 0.01 --n-augment 5
sbatch -p besteffort run.sh ANN --data higgs-uci --batch-size 1024 --width 0.03 --n-augment 2
sbatch -p besteffort run.sh ANN --data higgs-uci --batch-size 1024 --width 0.03 --n-augment 5
sbatch -p besteffort run.sh ANN --data higgs-uci --batch-size 1024 --width 0.05 --n-augment 2
sbatch -p besteffort run.sh ANN --data higgs-uci --batch-size 1024 --width 0.05 --n-augment 5

# # Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.0
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.001
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.01
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 100.0


# # Pivot Adversarial Network vs width
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data higgs-uci --batch-size 1024 --width 0.05 --trade-off 1.0


# ============================================================================
#	HIGGS GEANT 4
# ============================================================================

# # Standard Neural Net
sbatch -p besteffort run.sh NN --data higgs-geant --batch-size 1024


# # Tangent Propagation vs trade-off
sbatch -p besteffort run.sh TP --data higgs-geant --batch-size 1024 --trade-off 0.0
sbatch -p besteffort run.sh TP --data higgs-geant --batch-size 1024 --trade-off 0.001
sbatch -p besteffort run.sh TP --data higgs-geant --batch-size 1024 --trade-off 0.01
sbatch -p besteffort run.sh TP --data higgs-geant --batch-size 1024 --trade-off 0.1
sbatch -p besteffort run.sh TP --data higgs-geant --batch-size 1024 --trade-off 1.0
sbatch -p besteffort run.sh TP --data higgs-geant --batch-size 1024 --trade-off 10.0
sbatch -p besteffort run.sh TP --data higgs-geant --batch-size 1024 --trade-off 100.0


# # Data augmentation vs width
sbatch -p besteffort run.sh ANN --data higgs-geant --batch-size 1024 --width 0.01 --n-augment 2
sbatch -p besteffort run.sh ANN --data higgs-geant --batch-size 1024 --width 0.01 --n-augment 5
sbatch -p besteffort run.sh ANN --data higgs-geant --batch-size 1024 --width 0.03 --n-augment 2
sbatch -p besteffort run.sh ANN --data higgs-geant --batch-size 1024 --width 0.03 --n-augment 5
sbatch -p besteffort run.sh ANN --data higgs-geant --batch-size 1024 --width 0.05 --n-augment 2
sbatch -p besteffort run.sh ANN --data higgs-geant --batch-size 1024 --width 0.05 --n-augment 5

# # Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.0
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.001
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.01
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 100.0


# # Pivot Adversarial Network vs width
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch -p besteffort run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 1.0
