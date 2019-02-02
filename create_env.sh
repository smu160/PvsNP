#!/usr/bin/env bash

printf "\e[36;1mCreating an environment for PvsNP... \e[0m \n"
conda env create -f environment.yml
if [ $? -eq 0 ]; then
    printf "\e[36;1mNew conda env creation SUCCESSFUL \e[0m \n"
else
    printf "\e[36;1mNew conda env creation FAILED \e[0m \n"
    exit 1
fi

printf "\e[36;1mActivating newly created conda environment... \e[0m \n"
source activate pvsnp_env
if [ $? -eq 0 ]; then
    printf "\e[36;1mSUCCESS \e[0m \n"
else
    printf "\e[36;1mpvsnp_env activation FAILED \e[0m \n"
    exit 1
fi

printf "\e[36;1mEnvironment creation and dependency installation complete! \e[0m \n"
