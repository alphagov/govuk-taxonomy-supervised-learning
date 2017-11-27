# govuk-taxonomy-supervised-machine-learning

Automatically tag content tags to taxons using machine learning algorithms.

## Getting the data

The following data files are used in this project.

|Name|Location|Description|Size|Updated|
|---|---|---|---|---|
|raw_taxons.json|s3://buod-govuk-taxonomy-supervised-learning/raw_taxons.json|List of taxons|1.1MB|2017-11-22|
|raw_content.json.gz|s3://buod-govuk-taxonomy-supervised-learning/raw_content.json|Content of GOV.UK (zipped to save space)|224MB|2017-11-22|

The files `raw_content.json` and `raw_taxons.json` can be downloaded from the S3 bucket where they are stored. To do this, navigate to the repo's `data` folder from the command line and download the required files using the command: `aws s3 cp s3://buod-govuk-taxonomy-supervised-learning/<file> .` (the dot ensures they are downloaded into the current folder that you have navigated to). The file `raw_content.json` is stored as a `.gz` file and will need to be unzipped with `gunzip raw_content.json`.

