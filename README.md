Hen Lab
======
# <img src="neurons_pic.png" >

* [Image source](https://www.flickr.com/photos/pennstatelive/26133296018/in/photolist-FPiYrh-ayGQme-oh3wgN-3Y9rPw-7EbTfz-5uoCvQ-6RFagT-cXaYiW-26saMb5-cXaYaf-SjW3Ej-5dopRY-577km2-SAAVs-5giagf-pUE6zU-7aC9da-cUjTYd-b3V8ue-4kMWWf-nxMXcN-af2bm-7vBoAp-2WE91-GL7hrL-8eisF6-NvaH2-8d8VPZ-654maj-9TSa47-8P1nS9-2kLAC-5TpNDj-7ViV3Q-rrREQF-GEMqn7-5H4b4k-phmhkw-dZBAtA-58VDW-REmihs-339Dq-dXKxjW-8weapo-9qsdEj-9UwZT-84RE2M-T2fmWP-e9UeMM-9UwZJ)

This code base was written with the intention of serving as a toolbox for:

1. Statistical Testing of neuron classification by behavior.
2. Graph Theoretical Analysis of networks of neurons.
3. Data Visualization tool -- see the "full picture" of the data garnered from experiments with integration of calcium imaging movies, behavioral videos, and data processing results.
4. Deconvolution of calcium imaging data. (Deprecated)

For any feature requests, [create an issue](https://help.github.com/articles/creating-an-issue/).

## Getting Started

### Docker

1. Download and install [Docker](https://www.docker.com/get-started)
2. Build the Docker image:
```bash
docker build --build-arg USERNAME=your_GitHub_username --build-arg PASSWORD=your_GitHub_password . -t jupyter
```
3. Use the image to run a container.
```bash
docker run -it -p 8888:8888 jupyter
```
If you need to mount data to the container, then use the following command:
```bash
docker run -it -p 8888:8888 -v source_directory:target_directory jupyter
```
4. You should see something along the lines of:
```bash
...
    Or copy and paste one of these URLs:
        http://(2958ngdf42t or 127.0.0.1):8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Now open your browser window and go to the URL that was created for you.

### Installation on Mac or Linux (Python 3.x)

* Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) (Python 3.6)

```bash
git clone https://github.com/jaberry/Hen_Lab.git
cd Hen_Lab
bash create_env.sh
```

## Troubleshooting

[Create an issue] (https://help.github.com/articles/creating-an-issue/)
For members of the Hen Lab, feel free to message Saveliy Yusufov via Slack.

## Authors

* **Jack Berry** - [jaberry](https://github.com/jaberry)
* **Saveliy Yusufov** - [smu160](https://github.com/smu160)

## Why Python?
- [Eight Advantages of Python Over Matlab](http://phillipmfeldman.org/Python/Advantages_of_Python_Over_Matlab.html)
- [Interactive notebooks: Sharing the code](https://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261)
- [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)
- [Food for Thought](https://www.theatlantic.com/science/archive/2018/04/the-scientific-paper-is-obsolete/556676/)
