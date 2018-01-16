# govuk-taxonomy-supervised-machine-learning

Automatically tag content tags to taxons using machine learning algorithms.

## Requirements

* Python 3.4.6
* See [requirements.txt](requirements.txt) for python dependencies.
* The amazon web services (AWS) command line interface (CLI), see below.

## Setting environment variables

A number of environment variables need to be set before running the cleaning scripts on your system:

|ENV VAR|Description|Nominal value|
|---|---|---|
|DATADIR|Path to the directory storing the data|`./data` (relative to the root of the repository -- you may need to set an absolute path)|
|LOGGING_CONFIG|Path to the logging configuration file|`./python/logging.conf` (relative to the root of the repository -- you may need to set an absolute path)|
|S3BUCKET|Path of the S3 bucket in which the data are stored.|s3://buod-govuk-taxonomy-supervised-learning|

## Preparing your python environment

The Makefile assumes that the `python3` command is pointing to the correct distribution of python, which was 3.4.6 in development. To install the correct package dependencies run `make init` from the project root.

## Getting the data

The following data files are used in this project.

|Name|Location|Description|Size|
|---|---|---|---|
|raw_taxons.json|s3://buod-govuk-taxonomy-supervised-learning/raw_taxons.json|List of taxons|1.1MB|
|raw_content.json.gz|s3://buod-govuk-taxonomy-supervised-learning/raw_content.json|Content of GOV.UK (zipped to save space)|224MB|
|document_type_group_lookup.json|s3://buod-govuk-taxonomy-supervised-learning/document_type_group_lookup.json|Lookup table for document type groups|2KB|

You will need access to the `s3://govuk-taxonomy-supervised-learning` S3 bucket to access these files, which requires programmatic access to Amazon Web Services (AWS). Once a key has been created for you, you will need to install the AWS command line interface (CLI), which can be done with:

`brew install aws`

Then configure an AWS CLI profile with:

`aws configure --profile gds-data`

When asked to set a default region set `eu-west-2` (London), and default format `json`.

If these files do not exist in DATADIR, they will be created by the Makefile by running `make`.

### Manual download

If necessary, files `raw_content.json`, `raw_taxons.json`, and `document_type_group_lookup.json` can be downloaded manually from the S3 bucket using the `aws s3 cp` command. This command works exactly like the bash `cp`, e.g.: to copy a file from the s3 bucket to your local machine:

```
aws s3 cp s3://buod-govuk-taxonomy-supervised-learning/<file> <local file>
```

To copy a local file to the s3 bucket, use:

```
aws s3 cp s3://buod-govuk-taxonomy-supervised-learning/<file> <local file>
```

Assuming you have set your `S3BUCKET` env variable, you can also just do:

```
aws s3 cp $S3BUCKET/<file> <local file>
```

__NOTE: The s3 bucket is version controlled, so if writing to the bucket, you do not need to rename the files to reflect the date files were produced. Just overwrite the existing file with the same filename.__

Some files are stored compressed like `raw_content.json.gz`. Do not decompress these files, as the data cleaning scripts will load the data from the compressed files automatically.

## Running the cleaning scripts

After setting environment variables and installing the AWS CLI, running `make` will download the data and launch the cleaning scripts in order. The following files are created by the various cleaning scripts:

|Filename (data/)|produced by (python/)|
|---|---|
|clean_taxons.csv|clean_taxons.py|
|clean_content.csv|clean_content.py|
|untagged_content.csv|clean_content.py|
|empty_taxons.csv|create_labelled.py|
|labelled.csv|create_labelled.py|
|filtered.csv|create_labelled.py|
|old_taxons.csv|create_labelled.py|
|empty_taxons.csv|create_labelled.py|
|labelled_level1.csv|create_labelled.py|
|labelled_level2.csv|create_labelled.py|
|empty_taxons_not_world.csv|create_labelled.py|
|new_content.csv|create_new.py|
    
The following schematic describes the movement of data through the pipeline, and the role of each of the scripts.

![alt text](data_map.png)

The cleaned files are used by the python notebooks contained in `python/notebooks`.

## Jupyter notebooks

### Setting up a Jupyter kernel

You should use your virtualenv when running jupyter notebooks. Follow the following steps:

Install the ipython kernel module into your virtualenv
```{bash}
workon my-virtualenv-name  # activate your virtualenv, if you haven't already
pip install ipykernel
```

Now run the kernel "self-install" script:
```{bash}
python -m ipykernel install --user --name=my-virtualenv-name
```

Replacing the --name parameter as appropriate.

You should now be able to see your kernel in the IPython notebook menu: Kernel -> Change kernel and be able so switch to it (you may need to refresh the page before it appears in the list). IPython will remember which kernel to use for that notebook from then on.

### Notebooks
|Name|Activity|Data inputs|Data outputs|
|---|---|---|---|
|EDA-count-data|Read in and count data files|untagged_content.csv, clean_taxons.csv, clean_content.csv.gz, labelled.csv, filtered.csv, empty_taxons.csv, old_tags.csv|None|
|EDA-taxons|Descriptive analysis of taxon content overall, and according to level|labelled, filtered, taxons|level2taxons_concordant.csv, taggedtomorethan10taxons.csv|
|EDA-document-type|Descriptive analysis of content according to document type, over time|untagged, labelled, filtered, labelled_level1, labelled_level2|document_type_group_lookup.json|
|EDA-other-metadata|Descriptive analysis of content according to metadata types, over time|untagged, labelled, filtered, labelled_level1, labelled_level2|none|

## Machine learning notebooks (ML_notebooks)
|Name|Activity|Data inputs|
|---|---|---|
|CNN-allgovuk.ipynb|Convolutional Neural Network of tagged content using keras framework and pre-trained word embeddings|clean_content.csv.gz, clean_taxons.csv|
|SVM_allgovuk.ipynb|Support vector machine of tagged content||
|TPOT_allgovuk.ipynb|Genetic algorithm to select optimal algorithm and hyperparameters||

## Archived notebooks
|Name|Activity|Data inputs|Data outputs|
|---|---|---|---|
|EDA|Exploratory data analysis|untagged_content.csv, clean_taxons.csv, clean_content.csv.gz|None|
|clean_content.ipynb|Development of steps to process raw content data into formats for use in EDA and modelling. These are now used in clean_content.py, which is called by the Makefile|||
|explore_content_dupes.ipynb|Understand duplicates in gov.uk content items|raw_content.json, clean_content.csv|None|

## Logging

The default logging configuration used by the data transformation pipeline (set in `./python/`) will do the following things:

* Write a simple log to stdout (console) at `INFO` level
* Write a more detailed log to a file at `DEBUG` level (by default `/tmp/govuk-taxonomy-supervised-learning.log`).

## Setting up Tensorflow/Keras on GPU backed instances on AWS

Setting up GPU-backed instances on AWS is greatly facilitated by using [databox](https://github.com/ukgovdatascience/databox). Currently the features required to create a deep learning instances are in [Pull Request 31](https://github.com/ukgovdatascience/databox/pull/31). Once these are merged into master, operate databox from master, but for now you will need to `git  checkout feature/playbook_argument`. Once you have databox and all its dependencies installed, the following command will instantiate an instance prepared for Deep Learning on AWS:

```
./databox.sh -a ami-1812bb61 -r eu-west-1 -i p2.xlarge -s snap-04eb15f2e4faee97a -p playbooks/govuk-taxonomy-supervised-learning.yml up
```

The arguments are explained in the table below:

|Argument|Value|Description|
|---|---|---|
|-a|ami-1812bb61|The Conda based Amazon Machine Image. Other options are explained by [amazon](https://aws.amazon.com/blogs/machine-learning/new-aws-deep-learning-amis-for-machine-learning-practitioners/).|
|-i|p2.xlarge|This is the smallest of the Deep Learning instance types. More information is available [here](https://aws.amazon.com/ec2/instance-types/). Note that the Deep Learning AMIs may not work with the newer p3 GPU instances.|
|-s|snap-04eb15f2e4faee97a|The id of the snapshot containing the taxonomy data. This can be checked at the [snapshot](https://eu-west-1.console.aws.amazon.com/ec2/v2/home?region=eu-west-1#Snapshots:sort=snapshotId)|
|-p|playbooks/govuk-taxonomy-supervised-learning.yml|The ansible playbook describing deployment tasks required on to setup the instance.|
|-r|eu-west-1|The region in which the instance will be deployed. At present this must be set to `eu-west-1` (Ireland) as some deep learning instances are not available in the `eu-west-2` (London) zone, and the snapshot is currently in `eu-west-1` (although could be copied elsewhere.|

Once the instance has instantiated, you will need to run the following commands:

* SSH tunnel into the instance with `ssh -L localhost:8888:localhost:8888 ubuntu@<databox IP>`
* Activate the tensorflow_p36 environment and run `jupyter notebook` on the instance:
```
source activate tensorflow_p36
jupyter notebook
```
This will set up a notebook server, for which you will be provided a link in the console.
* Log in to notebook server on your __local__ machine by copying the link generated on the server into a browser. This will give you access to jupyter notebooks on your local machine.

### Tensorboard

* To run tensorboard ensure that the tensorboard callback has been enabled in the model, then log into the instance again in a new terminal creating a new tunnel with `ssh -L localhost:6006:localhost:6006 ubuntu@<databox IP>`. 
* Activate the `tensorflow_p36` environment with `source activate tensorflow_p36`.
* Run `tensorboard --log_dir=<path to logging>`.
* Open a browser on your local machine and navigate to <https://localhost:6006> to access the tensorboard sever.

### Check that the GPU is doing the work

* Ensure that your model is running on the instance GPU by running `nvidia-smi` in a new terminal on the instance (you can run this repeatedly with `watch -n 10 nvidia-smi` to update every 10 seconds).

