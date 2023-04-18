1) Be sure you don't have any 'active' images or containers:
1.1) docker rm -f -v $(docker ps -aq)
1.2) docker rmi -f  $(docker images -q)
1.3) docker volume rm $(docker volume ls -q)

2) Create conda environment:

2.1) conda create -n mosquinha python=3.10

2.2) conda activate mosquinha

2.3) Install requirements from the command line: pip install -r requirements.txt


3) Change the parameters in the file conf.yaml, in case you want, except the outputs parameter

4) Run it: python main.py -c conf/conf.yaml -r [state] -i [image] --preproc [preprocessing method]
For example: python main.py -c conf/conf.yaml -r full -i in/SF14/day1_low10.bmp
For 'state' parameter you can choose:
- 'preproc': it only does image preprocessing
- 'prepare': it does Preprocessing -> Train -> Test -> Performance plots and metrics
- 'classify': it classifies the respective image. You must have a trained model first and point to it with the -m flag
- 'full': it does everything from preprocessing to classification

Obs: The results from training and testing are saved in /out/outputs.txt

In conf.yaml we can choose the parameters:
- optim: 'Adam'
- model: {'densenet', 'resnet', 'efficientnet'}
- epochs: Any integer greater than 0
- batch: 16
- lr: !!float 5e-4
- raw: 'in/'
- preproc: {'skip', 'gaussian', 'gauss_threshold', 'median', 'bilateral', 'unsharp'}

