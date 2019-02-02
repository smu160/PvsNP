# Get and use latest LTS release of Ubuntu
FROM ubuntu:18.04

# Updating Ubuntu package & install wget, bzip2, and git
RUN apt-get update && apt-get install -y \
  wget \
  bzip2 \
  git

COPY ./ /

# Install Anaconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh
RUN bash Miniconda3-4.5.11-Linux-x86_64.sh -b
RUN rm Miniconda3-4.5.11-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/miniconda3/bin:$PATH

# Install required packages
RUN conda env update -n root --file environment.yml

# Jupyter listens port: 8888
EXPOSE 8888

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
