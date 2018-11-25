# Get and use latest LTS release of Ubuntu
FROM ubuntu:latest

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade

# Adding wget and bzip2
RUN apt-get install -y wget bzip2 git

# Install Anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
RUN bash Anaconda3-5.3.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.3.1-Linux-x86_64.sh 

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

# Configure access to Jupyter
RUN mkdir /opt/notebooks

# Clone repo
RUN git clone https://<Username>:<Password>@github.com/jaberry/Hen_Lab.git

# Jupyter listens port: 8888
EXPOSE 8888

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
