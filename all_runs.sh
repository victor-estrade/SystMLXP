#!/bin/sh


# ============================================================================
#	MNIST
# ============================================================================

# # Standard Neural Net
sbatch run.sh NN --data mnist --batch-size 128 --width 5 --trade-off 1.0


# # Tangent Propagation vs trade-off
sbatch run.sh TP --data mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch run.sh TP --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh TP --data mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch run.sh TP --data mnist --batch-size 128 --width 5 --trade-off 100.0


# # Data perturbation vs width
sbatch run.sh NNDA --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh NNDA --data mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch run.sh NNDA --data mnist --batch-size 128 --width 15 --trade-off 1.0


# # Data augmentation vs width
sbatch run.sh NNA --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh NNA --data mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch run.sh NNA --data mnist --batch-size 128 --width 15 --trade-off 1.0


# # Pivot Adversarial Network vs trade-off
sbatch run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 100.0


# # Pivot Adversarial Network vs width
sbatch run.sh PAN --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh PAN --data mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch run.sh PAN --data mnist --batch-size 128 --width 15 --trade-off 1.0

# ============================================================================
#	FASHION MNIST
# ============================================================================

# Standard Neural Net
sbatch run.sh NN --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0


# Tangent Propagation vs trade-off
sbatch run.sh TP --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch run.sh TP --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh TP --data fashion-mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch run.sh TP --data fashion-mnist --batch-size 128 --width 5 --trade-off 100.0


# Data augmentation vs width
sbatch run.sh NNA --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh NNA --data fashion-mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch run.sh NNA --data fashion-mnist --batch-size 128 --width 15 --trade-off 1.0

# Data perturbation vs width
sbatch run.sh NNDA --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh NNDA --data fashion-mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch run.sh NNDA --data fashion-mnist --batch-size 128 --width 15 --trade-off 1.0

# Pivot Adversarial Network vs trade-off
sbatch run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 100.0


# Pivot Adversarial Network vs width
sbatch run.sh PAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch run.sh PAN --data fashion-mnist --batch-size 128 --width 10 --trade-off 1.0
sbatch run.sh PAN --data fashion-mnist --batch-size 128 --width 15 --trade-off 1.0


# ============================================================================
#	HIGGS UCI
# ============================================================================

# # Standard Neural Net
sbatch run.sh NN --data higgs-uci --batch-size 1024 --width 0.01 --trade-off 1.0


# # Tangent Propagation vs trade-off
sbatch run.sh TP --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch run.sh TP --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh TP --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch run.sh TP --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 100.0


# # Data perturbation vs width
sbatch run.sh NNDA --data higgs-uci --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch run.sh NNDA --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh NNDA --data higgs-uci --batch-size 1024 --width 0.05 --trade-off 1.0

# # Data augmentation vs width
sbatch run.sh NNA --data higgs-uci --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch run.sh NNA --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh NNA --data higgs-uci --batch-size 1024 --width 0.05 --trade-off 1.0

# # Pivot Adversarial Network vs trade-off
sbatch run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 100.0


# # Pivot Adversarial Network vs width
sbatch run.sh PAN --data higgs-uci --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch run.sh PAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh PAN --data higgs-uci --batch-size 1024 --width 0.05 --trade-off 1.0


# ============================================================================
#	HIGGS GEANT 4
# ============================================================================

# # Standard Neural Net
sbatch run.sh NN --data higgs-geant --batch-size 1024 --width 0.01 --trade-off 1.0


# # Tangent Propagation vs trade-off
sbatch run.sh TP --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch run.sh TP --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh TP --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch run.sh TP --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 100.0


# # Data perturbation vs width
sbatch run.sh NNDA --data higgs-geant --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch run.sh NNDA --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh NNDA --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 1.0


# # Data augmentation vs width
sbatch run.sh NNA --data higgs-geant --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch run.sh NNA --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh NNA --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 1.0

# # Pivot Adversarial Network vs trade-off
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 100.0


# # Pivot Adversarial Network vs width
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.01 --trade-off 1.0
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 1.0
