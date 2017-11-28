# govuk-taxonomy-supervised-machine-learning

Automatically tag content tags to taxons using machine learning algorithms.

## Requirements

* Python 3.4.3
* See [requirements.txt](requirements.txt) for python dependencies

## Getting the data

The following data files are used in this project.

|Name|Location|Description|Size|Updated|
|---|---|---|---|---|
|raw_taxons.json|s3://buod-govuk-taxonomy-supervised-learning/raw_taxons.json|List of taxons|1.1MB|2017-11-22|
|raw_content.json.gz|s3://buod-govuk-taxonomy-supervised-learning/raw_content.json|Content of GOV.UK (zipped to save space)|224MB|2017-11-22|

The files `raw_content.json` and `raw_taxons.json` can be downloaded from the S3 bucket where they are stored using the `aws s3 cp` command. This command works exactly like the bash `cp`, e.g.: to copy a file from the s3 bucket to your local machine:

```
aws s3 cp s3://buod-govuk-taxonomy-supervised-learning/<file> <local file>
```

To copy a local file to the s3 bucket, use:

```
aws s3 cp s3://buod-govuk-taxonomy-supervised-learning/<file> <local file>
```

__NOTE: The s3 bucket is version controlled, so you do not need to rename the files to reflect the date files were produced. Just overwrite the existing file with the same filename.__

Some files are stored compressed like `raw_content.json.gz`. Do not decompress these files, as the data cleaning scripts will load the data from the compressed files automatically.

## Setting environment variables

A number of environment variables need to be set before running the cleaning scripts on your system:

|ENV VAR|Description|Nominal value|
|---|---|---|
|DATADIR|Path to the directory storing the data|`./data` (relative to the root of the repository -- you may need to set an absolute path)|
|LOGGING_CONFIG|Path to the logging configuration file|`./python/logging.conf` (relative to the root of the repository -- you may need to set an absolute path)|

## Running the cleaning scripts

There is a Makefile which can be run to execute the cleaning scripts. After downloading the data to an appropriate dir, and pointing at it with the env vars, you can install python dependencies by running `make init`.

Once complete, running `make` will launch the cleaning scripts, creating two files:

* data/clean_taxons.csv
* data/clean_content.csv

These cleaned files are used by the python notebooks contained in `python/notebooks`.

## Logging

The default logging configuration (set in `./python/`) will do the following things:

* Write a simple log to stdout (console) at `INFO` level
* Write a more detailed log to a file at `DEBUG` level (by default `/tmp/govuk-taxonomy-supervised-learning.log`).
