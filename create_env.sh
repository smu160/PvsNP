#!/usr/bin/env bash

printf "\e[36;1mNow updating conda's root env... \e[0m \n"
conda update -n root conda
if [ $? -eq 0 ]; then
    printf "\e[36;1mconda update SUCCESSFUL \e[0m \n"
else
    printf "\e[36;1mconda update FAILED \e[0m \n"
    exit 1
fi

printf "\e[36;1mNow updating Anaconda... \e[0m \n"
conda update --all
if [ $? -eq 0 ]; then
    printf "\e[36;1mconda update SUCCESSFUL \e[0m \n"
else
    printf "\e[36;1mconda update FAILED \e[0m \n"
    exit 1
fi

printf "\e[36;1mCreating an environment for Hen Lab code base... \e[0m \n"
conda create --name henlabenv anaconda
if [ $? -eq 0 ]; then
    printf "\e[36;1mNew conda env creation SUCCESSFUL \e[0m \n"
else
    printf "\e[36;1mNew conda env creation FAILED \e[0m \n"
    exit 1
fi

printf "\e[36;1mActivating newly created conda environment... \e[0m \n"
source activate henlabenv
if [ $? -eq 0 ]; then
    printf "\e[36;1mSUCCESS \e[0m \n"
else
    printf "\e[36;1mhenlabenv activation FAILED \e[0m \n"
    exit 1
fi

printf "\e[36;1mUpdating Anaconda once more... \e[0m \n"
conda update --all
if [ $? -eq 0 ]; then
    printf "\e[36;1mconda update SUCCESSFUL \e[0m \n"
else
    printf "\e[36;1mconda update FAILED \e[0m \n"
    exit 1
fi

printf "\e[36;1mEnvironment creation and dependency installation complete! \e[0m \n"
