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

Once downloaded, run the script at `src/dataset` called `crawler.py`, which will download all MIDI files into `res/dataset/raw`.