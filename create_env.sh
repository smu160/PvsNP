#!/usr/bin/env bash

printf "\e[36;1mNow updating conda... \e[0m \n"
conda update -n root conda

printf "\e[36;1mNow updating Anaconda... \e[0m \n"
conda update --all

printf "\e[36;1mCreating an environment for Hen Lab code base... \e[0m \n"
conda env create -f environment.yml 

printf "\e[36;1mActivating newly created conda environment... \e[0m \n"
source activate henlabenv

printf "\e[36;1mInstalling Node.js, requirement for plotly... \e[0m \n"
conda install -c conda-forge nodejs

printf "\e[36;1mInstalling plotly extension for JupyterLab... \e[0m \n"
jupyter labextension install @jupyterlab/plotly-extension

printf "\e[36;1mEnvironment creation and dependency installation complete! \e[0m \n"
