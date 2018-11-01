Hen Lab 
======
# <img src="neurons_pic.png" >

* [Image source](https://www.flickr.com/photos/pennstatelive/26133296018/in/photolist-FPiYrh-ayGQme-oh3wgN-3Y9rPw-7EbTfz-5uoCvQ-6RFagT-cXaYiW-26saMb5-cXaYaf-SjW3Ej-5dopRY-577km2-SAAVs-5giagf-pUE6zU-7aC9da-cUjTYd-b3V8ue-4kMWWf-nxMXcN-af2bm-7vBoAp-2WE91-GL7hrL-8eisF6-NvaH2-8d8VPZ-654maj-9TSa47-8P1nS9-2kLAC-5TpNDj-7ViV3Q-rrREQF-GEMqn7-5H4b4k-phmhkw-dZBAtA-58VDW-REmihs-339Dq-dXKxjW-8weapo-9qsdEj-9UwZT-84RE2M-T2fmWP-e9UeMM-9UwZJ)

This code base was written with the intention of serving as a toolbox for:

1. Signal Processing of neuron data collected via *in vivo* calcium imaging.
2. Statistical Testing of neuron classification by behavior.
3. Graph Theoretical Analysis of networks of neurons. 

For any feature requests, feel free to email Saveliy Yusufov at sy2685@columbia.edu

## Getting Started

### Installation on Mac or Linux (Python 3.x)

* Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 3.6) 

```bash
git clone https://github.com/jaberry/Hen_Lab.git
cd Hen_Lab
bash create_env.sh
```   

### Installation on Windows 

#### (Python 2.x)

* Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 2.7) We recommend telling conda to modify your PATH variable (it is a checkbox during Anaconda install, off by default).

* Launch an anaconda-enabled command prompt as follows: start --> Anaconda2 --> Anaconda Prompt

* Use conda to install git as follows: `conda install git`

```bash
git clone https://github.com/jaberry/Hen_Lab.git
cd Hen_Lab
conda create -n henlabenv anaconda python=3.6
conda activate henlabenv
```

#### (Python 3.x)

* Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 3.6) We recommend telling conda to modify your PATH variable (it is a checkbox during Anaconda install, off by default).

* Launch an anaconda-enabled command prompt as follows: start --> Anaconda3 --> Anaconda Prompt

* Use conda to install git as follows: `conda install git`

```bash
git clone https://github.com/jaberry/Hen_Lab.git
cd Hen_Lab
conda list --export > conda_packages.txt
conda create --name henlabenv --file conda_packages.txt
conda activate henlabenv
```


## Troubleshooting

For any issues, feel free to email Saveliy Yusufov at sy2685@columbia.edu

## Built With

* [pandas](http://pandas.pydata.org)
* [NetworkX](https://networkx.github.io)
* [seaborn](http://seaborn.pydata.org)

## Authors

* **Jack Berry** - [jaberry](https://github.com/jaberry)
* **Saveliy Yusufov** - [smu160](https://github.com/smu160)

## Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)
