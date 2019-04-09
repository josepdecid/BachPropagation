# BachPropagation
Classical Music Generator

## Installing requirements

For this project, we will use `Python3.7`, managing the necessary Python packages using 
[Pipenv: Python Dev Workflow for Humans](https://pipenv.readthedocs.io/en/latest/).
You can install it following the steps shown at their installation section.

To install the requirements listed in the Pipfile, run:
```bash
$ pipenv install
```

To add a new package to the project:
```bash
$ pipenv install <package>
```

To activate the python shell, run the following command, which will spawn a new shell subprocess, which can be deactivated by using `exit`.
````bash
$ pipenv shell
````

## Download data

First of all, to work with this project, we are going to need a Environment variable pointing to our root.
E.g. if our project is at /home/projects/bachpropagation, just add to your`favorite bash profile:

```bash
export BACHPROPAGATION_ROOT_PATH=/home/projects/bachpropagation
```

To download the raw midi files, run the script at `src/dataset` called `crawler.py`,
which will download all MIDI files into `res/dataset/raw`. Then, to make them easier to process, we will convert those
into CSV files by running `src/dataset/parser.py` that will write the processed files into `res/dataset/processed`.

## Training

To train the model, run `main.py`, which will automatically read and instantiate the Datasets and start the training process.
For hyperparameter tuning, change some default folders configuration... visit `constants.py`.

*⚠️ Warning:* If not using PyCharm or any other powerful IDE, you'll have to specify the Python sources path manually:

```bash
# Standard running command
$ python main.py

# Extend Python path
$ PYTHONPATH="${PYTHONPATH}:src" python main.py
```

For each iteration it's going to print an output like the following:

```plain
Epoch 1: 100%|███████████| 10/10 [00:15<00:00,  1.88s/it]
Generator loss: -0.684159 | Discriminator loss: -1.386941
INFO:root:Creating Sample 1
INFO:root:Converting to CSV...
INFO:root:Parsing note data...
INFO:root:Writing MIDI file at ~/bachpropagation/res/results/
```

There is an option to visualize the training evolution with [Visdom](https://github.com/facebookresearch/visdom),
already installed in the *pipenv* dev-dependencies (running `pipenv install --dev`).
To run the Visdom server just write the following comand in the console (you can also specify custom host, port and other configurations):

```bash
$ visdom
```

Be sure to have the Visdom server running before training the model, then run the model with the visualization flag:

```bash
$ python model.py --viz
```