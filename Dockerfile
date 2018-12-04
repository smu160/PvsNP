# Get and use latest LTS release of Ubuntu
FROM ubuntu:latest

ARG USERNAME=username
ARG PASSWORD=password

# Updating Ubuntu package & install wget, bzip2, git, and g++
RUN apt-get update && apt-get install -y \
  wget \
  bzip2 \
  git \
  g++

# Install Anaconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh
RUN bash Miniconda3-4.5.11-Linux-x86_64.sh -b
RUN rm Miniconda3-4.5.11-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/miniconda3/bin:$PATH

# Updating miniconda3 packages
RUN conda install Cython \
  numpy \
  matplotlib \
  pandas \
  networkx \
  jupyter \
  scikit-learn \
  seaborn

# Clone repo
RUN git clone https://$USERNAME:$PASSWORD@github.com/jaberry/Hen_Lab.git

# Clone OASIS repo
# RUN git clone https://github.com/j-friedrich/OASIS.git

# WORKDIR "OASIS"

# RUN python setup.py build_ext --inplace && python setup.py clean --all

# WORKDIR "/"

# Jupyter listens port: 8888
EXPOSE 8888

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
