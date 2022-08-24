# serverless-pypi

An AWS Lambda implementation for the PyPI protocols.

Python packaging is fantastic; however, challenges arise when you need to use standard PyPI (e.g. - pip) mechanms in CI/CD but you need to manage private projects. In this situation, you are generally limited to implementing complex open-source solutions (e.g. - warehouse, devpi, etc.) or paying for expensive commercial solutions (e.g. - Artifactory).

`serverless-pypi` is designed to largely elimiate these challenges. It is an AWS Lambda function that secures your private packages while fully mirroring *any* underlying repository, and it does all of this with **only the Lambda function itself and an S3 bucket**.

`serverless-pypi`:
1. Implements the PyPI simple JSON and HTML protocols and PyPI's upload protocol
2. Pull-through mirrors a base repository (https://pypi.org/simple by default; this may be another private repository if you wish)
3. Allows for localized upload of private packages; private packages with the same name as mirrored packages will override the mirror.
4. Manages users for both download and upload roles

## Performance
`serverless-pypi` is *fast* for `pip install` requests, as these requests are eventually redirected to AWS S3 presigned URLs. This is particularly true when you are accessing `serverless-pypi` within AWS itself.

Uploads using `twine` are still quite fast, but since these must be processed directly by the Lambda function will be slower than downloads. New packages uploaded will be made availabe within a few seconds. New files uploaded to an existing package are available immediately.

## Installation
`serverless-pypi` can be obtained from the `lambdalambdalambda` repo as a fully built Lamba package or can be cloned and built via `python setup.py ldist`.

### AWS Deployment
`serverless-pypi` may be deployed in AWS in the following ways:
1.  As a stand-alone Lambda function utilizing a [Lambda function URL](https://docs.aws.amazon.com/lambda/latest/dg/lambda-urls.html).
2. Fronted by an API Gateway, using either REST or HTTP.
3. Fronted by an AWS ALB.
4. As a Lambda@Edge function.

### Lambda settings
The Python 3.9 runtime is required.

We recommend that you provide at least 1536GB of memory to ensure speedy responses.

If you are deploying `serverless-pypi` stand-alone, you will need to provision a Lambda Function URL. If you wish to throttle invocations, limit the function's cconcurrency.

### Environment Variables
| Variable | Required | Description | Default |
| - | - | - | - |
| API_GATEWAY_BASE_PATH | N | Sets the base path for the Lambda function. Only applicable if this is fronted by an AWS API Gateway. | / |
| BUCKET | Y | The AWS S3 bucket that is used to store the PyPI information. | |
| LOGGING_LEVEL | N | Sets the logging level for the Lambda function | INFO |
| MIRROR_INDEX_URL | N | The url underlying PyPI repository to mirror. This may contain credentialing information. | https://pypi.org/simple/ |
| REPO_BASE_PREFIX | N | The prefix to use in the S3 bucket. | "" |

### IAM Permissions
| Permission | Resource | Note |
| - | - | - |
| s3:GetObject | {BUCKET}/{REPO_BASE_PREFIX}/* | Retrieval of stored indexes, packages and users |
| s3:PutObject | {BUCKET}/{REPO_BASE_PREFIX}/* | Storage of indexes, packages and users |
| s3:ListBucket | {BUCKET}/{REPO_BASE_PREFIX}/* | Listing of stored private packages and users |
| lambda:Invoke | itself | Allows automated reindexing when a new package is uploaded |
| lambda:UpdateFunctionConfiguration | itself | Allows for forced restart of the function when reindexing and putting/removing users |

> Note - additional permissions will be required based upon you deployment method.

## Using `serverless-pypi`

### `pip install` or equivalent
The repository base URI is `/simple/`. Depending on your deployment method you will need to add this to the base URL of the deployment (e.g. - for stand-alone deployment, this will be the Lambda Function URL).

For example, to `pip install simplejson` you would:
```sh
pip install --index-url https://{my_user}:{my_password}@{my_lambda_function_url}/simple simplejson
```

## Uploading private packages using twine
`serverless-pypi` will automatically create a new project for the first package file uploaded for the project. This has been fully tested with `twine`; if you use a different upload method modify accordingly.

Uploads are `POST`ed to the root path of the repository.

For example, uploading the `foobar` wheel using `twine` would look like:
```sh
twine upload --repository-url "https://{my_lambda_function_url}/" --username {my_user} --password {my_password} foobar-0.0.1-py3-non-any-whl
```


## Managing `serverless-pypi`

### Users
`serverless-pypi` maintains its own, internal username/password database in order to conform to PyPI's HTTP Basic authentication requirements. There is no public, anonymous access.

Two user types are supported: users that are able to read the repository (e.g. - `pip` users) and users that can read and write the repository (e.g. - `twine` users)

The usernames themselves are one-way hashed (using `sha256`), and the passwords are hashed using `pbkdf2_sha512`. This provides an irreversable mechanism to securely store user information.

Managing users requries directly invoking the deployed Lambda function; invocation can be either an event or request/response invocation, although the latter will only report errors.

| Action | Description | Event Payload |
| - | - | - |
| Put user | Adds/updates a user | ```{"putUser": {"username": "my_user", "password": "my_password", "upload": [true/false]}}``` |
| Remove user | Removes a user | ```{"removeUser": "my_user"}```

### Keeping the Lambda function warm
Startup of the Lambda function is somewhat expensive since the total index (~8MB) is loaded and parsed into memory; therefore, you may desire to keep the Lambda function "warm" to make sure that it responds promptly to your `pip` and `twine` requests.

To do this, simply invoke the deployed Lambda function with the payload `"reheat"`.

### Reindexing
Since the underlying mirrored PyPI repository will change periodically, it is necessary to reindex you `serverless-pypi` repository periodically.

This is accomplished via any invocation (e.g. - an AWS EventBus Scheduled Event) that does not come from ALB, AWS Gateway, the Lambda's Function URL or Lambda@Edge, or does not match the user management or reheating invocation payloads. When an unknown payload is received, `serverless-pypi` will pull the mirror's master index, reindex your instance and restart your deployed Lambda function.

## Limitations
Uploads from `twine` to your `serverless-pypi` repository are limited to package files of 6MB or less.

> Note - you may, if you wish, directly upload larger packages to the correct location in the S3 bucket. If you choose to do this, you must replicate the key/file structure that `serverless-pypi` uses, and then force a reindex after your upload.

There are no limitations on package file sizes that are downloaded.
