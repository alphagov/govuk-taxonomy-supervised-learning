# govuk-taxonomy-supervised-machine-learning

Automatically tag content tags to taxons using machine learning algorithms.

## Getting the data

The files `content.json` and `taxons.json` can be downloaded from the S3 bucket `s3://buod-govuk-taxonomy-supervised-learning`. Note that this S3 bucket has version control, so a complete record of all versions is available in this bucket.

Once you have authenticated with amazon S3, these files can be downloaded with `aws s3 cp <file> s3://buod-govuk-taxonomy-supervised-learning`.

