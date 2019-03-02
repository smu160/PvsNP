# Get and use latest LTS release of Ubuntu
FROM ubuntu:18.04

# Update Ubuntu & install wget, bzip2. Then, install Miniconda.
RUN apt-get update && apt-get install -y wget bzip2 \
    && wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
    && bash Miniconda3-4.5.11-Linux-x86_64.sh -b \
    && rm Miniconda3-4.5.11-Linux-x86_64.sh \
    && rm -rf /var/lib/apt/lists/*

COPY ./ /

# Set path to conda
ENV PATH /root/miniconda3/bin:$PATH

# Install required packages
RUN conda env update -n root --file environment.yml

# Jupyter listens port: 8888
EXPOSE 8888

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
