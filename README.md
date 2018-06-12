Hen Lab 
======
# <img src="neurons_pic.png" >

* [Image source](https://www.flickr.com/photos/pennstatelive/26133296018/in/photolist-FPiYrh-ayGQme-oh3wgN-3Y9rPw-7EbTfz-5uoCvQ-6RFagT-cXaYiW-26saMb5-cXaYaf-SjW3Ej-5dopRY-577km2-SAAVs-5giagf-pUE6zU-7aC9da-cUjTYd-b3V8ue-4kMWWf-nxMXcN-af2bm-7vBoAp-2WE91-GL7hrL-8eisF6-NvaH2-8d8VPZ-654maj-9TSa47-8P1nS9-2kLAC-5TpNDj-7ViV3Q-rrREQF-GEMqn7-5H4b4k-phmhkw-dZBAtA-58VDW-REmihs-339Dq-dXKxjW-8weapo-9qsdEj-9UwZT-84RE2M-T2fmWP-e9UeMM-9UwZJ)

This code base was written with the intention of serving as a toolbox for:

1. Signal Processing of neuron data collected via *in vivo* calcium imaging.
2. Statistical Testing of neuron classification by behavior.
3. Graph Theoretical Analysis of networks of neurons. 

### Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)

For any feature requests, feel free to email Saveliy Yusufov at sy2685@columbia.edu

## Getting Started

### Installation on Mac or Linux (Python 3.x)

* Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 3.6) 

```bash
git clone https://github.com/jaberry/Hen_Lab.git
cd Hen_Lab
bash create_env.sh
```

**WARNING** This code base strives to utilize the latest version of Python as 
well as the latest version of all the dependencies. If you do not create a 
separate environment for using this code base, you are at risk of creating a 
massive headache for yourself. It behooves you to not install dependencies to 
the same environment Python that your operating system is using. In addition, if
you update your base environment to Python 3.x, while other software you use 
requires an older version of Python, you will be in a bad place. For your 
convenience, a Bash script that creates a conda environment with all required
dependencies was made available. You can run it by following the aforementioned
instructions.     

## Troubleshooting

Some features, e.g., permutation testing will only work on Unix based operating
 systems, i.e., Linux and MacOS. This issue will be soon addressed with the 
creation of a Dockerfile, for those of you who are using Windows. In the mean 
time, you have two choices: 

1. Use a machine that is running a Unix based operating system.
2. Install and use any Linux distro of your choice. For beginners, we recommend [Ubuntu](https://www.ubuntu.com/download/desktop).

## Built With

* [pandas](http://pandas.pydata.org)
* [NetworkX](https://networkx.github.io)
* [seaborn](http://seaborn.pydata.org)
* [plotly](https://plot.ly)

## Authors

* **Jack Berry** - [jaberry](https://github.com/jaberry)
* **Saveliy Yusufov** - [smu160](https://github.com/smu160)

## Acknowledgments

* Jessica Jimenez 
