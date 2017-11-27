#!/bin/sh

rsync -av /home/estrade/Bureau/PhD/SystML/mnist/*.sh titanic:/home/tao/vestrade/workspace/SystML/mnist
rsync -av /home/estrade/Bureau/PhD/SystML/mnist/*.py titanic:/home/tao/vestrade/workspace/SystML/mnist
rsync -av /home/estrade/Bureau/PhD/SystML/mnist/models/*.py titanic:/home/tao/vestrade/workspace/SystML/mnist/models/
rsync -av /home/estrade/Bureau/PhD/SystML/mnist/problem/*.py titanic:/home/tao/vestrade/workspace/SystML/mnist/problem/

# rsync -av /home/estrade/Bureau/PhD/SystML/mnist/*.ipynb titanic:/home/tao/vestrade/workspace/SystML/mnist
# rsync -av titanic:/home/tao/vestrade/workspace/SystML/mnist/*.ipynb /home/estrade/Bureau/PhD/SystML/mnist/
