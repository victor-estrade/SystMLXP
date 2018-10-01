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

# # Augmented Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh APAN --data mnist --batch-size 128 --width 5 --trade-off 0.0
sbatch -p besteffort run.sh APAN --data mnist --batch-size 128 --width 5 --trade-off 0.001
sbatch -p besteffort run.sh APAN --data mnist --batch-size 128 --width 5 --trade-off 0.01
sbatch -p besteffort run.sh APAN --data mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch -p besteffort run.sh APAN --data mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch -p besteffort run.sh APAN --data mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch -p besteffort run.sh APAN --data mnist --batch-size 128 --width 5 --trade-off 100.0

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

# Augmented Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh APAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.0
sbatch -p besteffort run.sh APAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.001
sbatch -p besteffort run.sh APAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.01
sbatch -p besteffort run.sh APAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 0.1
sbatch -p besteffort run.sh APAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 1.0
sbatch -p besteffort run.sh APAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 10.0
sbatch -p besteffort run.sh APAN --data fashion-mnist --batch-size 128 --width 5 --trade-off 100.0


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

# # Augmented Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh APAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.0
sbatch -p besteffort run.sh APAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.001
sbatch -p besteffort run.sh APAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.01
sbatch -p besteffort run.sh APAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch -p besteffort run.sh APAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch -p besteffort run.sh APAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch -p besteffort run.sh APAN --data higgs-uci --batch-size 1024 --width 0.03 --trade-off 100.0

# # Cascade Neural Net
sbatch -p besteffort run.sh NNC --data higgs-uci --batch-size 1024 --fraction-signal-to-keep 0.95
sbatch -p besteffort run.sh NNC --data higgs-uci --batch-size 1024 --fraction-signal-to-keep 0.75
sbatch -p besteffort run.sh NNC --data higgs-uci --batch-size 1024 --fraction-signal-to-keep 0.5

# # Gradient Boosting
sbatch -p besteffort run.sh GB --data higgs-uci --learning-rate 0.1

# ============================================================================
#	HIGGS GEANT 4
# ============================================================================

# # Standard Neural Net
sbatch -p besteffort run.sh NN --data higgs-geant --batch-size 1024

# # Standard Neural Net without skewed variables
sbatch -p besteffort run.sh BNN --data higgs-geant --batch-size 1024


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

# # Pivot Adversarial Network vs recovery-steps
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 1.0 --n-recovery-steps 5 --n-steps 10000
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 1.0 --n-recovery-steps 20 --n-steps 10000
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 1.0 --n-recovery-steps 50 --n-steps 10000

sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 10.0 --n-recovery-steps 5 --n-steps 10000
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 10.0 --n-recovery-steps 20 --n-steps 10000
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 10.0 --n-recovery-steps 50 --n-steps 10000

sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 100.0 --n-recovery-steps 5 --n-steps 10000
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 100.0 --n-recovery-steps 20 --n-steps 10000
sbatch run.sh PAN --data higgs-geant --batch-size 1024 --width 0.05 --trade-off 100.0 --n-recovery-steps 50 --n-steps 10000


# # Augmented Pivot Adversarial Network vs trade-off
sbatch -p besteffort run.sh APAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.0
sbatch -p besteffort run.sh APAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.001
sbatch -p besteffort run.sh APAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.01
sbatch -p besteffort run.sh APAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 0.1
sbatch -p besteffort run.sh APAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 1.0
sbatch -p besteffort run.sh APAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 10.0
sbatch -p besteffort run.sh APAN --data higgs-geant --batch-size 1024 --width 0.03 --trade-off 100.0


# # Cascade Neural Net
sbatch -p besteffort run.sh NNC --data higgs-geant --batch-size 1024 --fraction-signal-to-keep 0.95
sbatch -p besteffort run.sh NNC --data higgs-geant --batch-size 1024 --fraction-signal-to-keep 0.75
sbatch -p besteffort run.sh NNC --data higgs-geant --batch-size 1024 --fraction-signal-to-keep 0.5


# # Gradient Boosting
sbatch -p besteffort run.sh GB --data higgs-geant --learning-rate 0.1

# # Gradient Boosting without skewed variables
sbatch -p besteffort run.sh BGB --data higgs-geant --learning-rate 0.1
