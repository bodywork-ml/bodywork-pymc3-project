# Serving Uncertainty

Most Machine Learning (ML) models return a point-estimate of the most likely data label, given an instance of feature data. There are many scenarios, however, where a point-estimate is not enough - where there is need to understand the model's uncertainty in the prediction. For example, when assessing risk, or more generally, when making decisions to optimise some organisational-level cost (or utility) function. This need is particularly acute when the cost is a non-linear function of the variable you're trying to predict.

For these scenarios, 'traditional' statistical modelling can provide access to the distribution of predicted labels, but these approaches are hard to scale and built upon assumptions that are often invalidated by the data they're trying to model. Alternatively, it is possible to train additional ML models for predicting specific quantiles, through the use of [quantile loss functions](https://towardsdatascience.com/quantile-regression-from-linear-models-to-trees-to-deep-learning-af3738b527c3), but this requires training one new model for every quantile you need to predict, which is inefficient.

Half-way between statistics and ML we have probabilistic programming, rooted in the methods of Bayesian inference. This notebook demonstrates how to train such a predictive model using [PyMC3](https://docs.pymc.io) - a Probabilistic Programming Language (PPL) for Python. We will demonstrate how a single probabilistic program can be used to support requests for point-estimates, arbitrary uncertainty ranges, as well as entire distributions of predicted data labels, for a non-trivial regression task.

We will then demonstrate how to use [FastAPI](https://fastapi.tiangolo.com) to develop a web API service, that exposes a separate endpoint for each type of prediction request: point, interval and density. Finally, we will walk you through how to deploy the service to Kubernetes, using [Bodywork](https://github.com/bodywork-ml/bodywork-core).

All of the files used in this project can be found in the [bodywork-pymc3-project](https://github.com/bodywork-ml/bodywork-pymc3-project) repository on GitHub. You can use this repo, together with this guide, to train the model and then deploy the web API to a Kubernetes cluster. Alternatively, you can use this repo as a template for deploying your own machine learning projects. If you're new to Kubernetes, then don't worry - we've got you covered - read on.

## Machine Learning Lifecycle

<div align="center">
<img src="https://bodywork-media.s3.eu-west-2.amazonaws.com/bodywork-pymc3-project-lifecycle.png"/>
</div>

We are going to recommend that the model is trained using the code in the [train_model.ipynb](https://github.com/bodywork-ml/bodywork-pymc3-project/blob/main/train_model.ipynb) notebook, which will persist all ML build artefacts to cloud object storage (AWS S3). We will then use [Bodywork](https://github.com/bodywork-ml/bodywork-core) to deploy the web API defined in the [serve_model.py](https://github.com/bodywork-ml/bodywork-pymc3-project/blob/main/serve_model.py) module, directly from this GitHub repo. This process should begin as a manual one, and once confidence in this process is establish, re-training can be automated by using Bodywork to deploy a two-stage train-and-serve pipeline that runs on a schedule - e.g., as demonstrated [here](https://bodywork.readthedocs.io/en/latest/quickstart_ml_pipeline/).

## A (very) Quick Introduction to Bayesian Inference

Like statistical data analysis (and ML to some extent), the main aim of Bayesian inference is to infer the unknown parameters in a model of the observed data. For example, to test a hypotheses about the physical processes that lead to the observations. Bayesian inference deviates from traditional statistics - on a practical level - because it explicitly integrates prior knowledge regarding the uncertainty of the model parameters, into the statistical inference process. To this end, Bayesian inference focuses on the posterior distribution,

$$
p(\Theta | X) = \frac{p(X | \Theta) \cdot p(\Theta)}{p(X)}
$$

Where,

- $\Theta$ is the vector of unknown model parameters, that we wish to estimate;
- $X$ is the vector of observed data;
- $p(X | \Theta)$ is the likelihood function, that models the probability of observing the data for a fixed choice of parameters; and,
- $p(\Theta)$ is the prior distribution of the model parameters.

The unknown model parameters are not limited to regression coefficients - Deep Neural Networks (DNNs) can be trained using Bayesian inference and PPLs, as an alternative to gradient descent - e.g., see the article by [Thomas Wiecki](https://twiecki.io/blog/2016/06/01/bayesian-deep-learning/).

If you're interested in learning more about Bayesian data analysis and inference, then an **excellent** (inspirational) and practical introduction is [Statistical Rethinking by Richard McElreath](https://xcelab.net/rm/statistical-rethinking/). For a more theoretical treatment try [Bayesian Data Analysis by Gelman & co.](http://www.stat.columbia.edu/~gelman/book/). If you're curious, read on!

## Running the Project Locally

To be able to run everything discussed below, clone the GitHub repo, create a new virtual environment and install the required packages,

```text
$ git clone https://github.com/bodywork-ml/bodywork-pymc3-project.git
$ cd bodywork-pymc3-project
$ python3.8 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

### Getting Started with Kubernetes

If you have never worked with Kubernetes before, then please don't stop here. We have written a guide to [Getting Started with Kubernetes for MLOps](https://bodywork.readthedocs.io/en/latest/kubernetes/#getting-started-with-kubernetes), that will explain the basic concepts and have you up-and-running with a single-node cluster on your machine, in under 10 minutes.

Should you want to deploy to a cloud-based cluster in the future, you need only to follow the same steps while pointing to your new cluster. This is one of the key advantages of Kubernetes - you can test locally with confidence that your production deployments will behave in the same way.

## Training the Model using PyMC3

A complete Bayesian modelling workflow is executed in details, within [train_model.ipynb](https://github.com/bodywork-ml/bodywork-pymc3-project/blob/main/train_model.ipynb). We summarise the steps in this notebook as follows,

- TODO - describe synthetic data
- TODO - describe model
- TODO - demonstrate predictions.

## Engineering the Web API using FastAPI

The ultimate aim of this tutorial, is to serve predictions from a probabilistic program, via a web API with multiple endpoints. This is achieved in a Python module we’ve named [serve_model.py](https://github.com/bodywork-ml/bodywork-pymc3-project/blob/main/serve_model.py), parts of which will be reproduced below.

This module loads the trained model that is persisted to cloud object storage when [train_model.ipynb](https://github.com/bodywork-ml/bodywork-pymc3-project/blob/main/train_model.ipynb) is run. Then, it configures [FastAPI](https://fastapi.tiangolo.com) to start a server with HTTP endpoints at:

- **/predict/v1.0.0/point** - for return point-estimates.
- **/predict/v1.0.0/interval** - for returning highest density intervals.
- **/predict/v1.0.0/density** - for returning the probability density (in discrete bins).

These endpoints are defined in the following functions (refer to [serve_model.py](https://github.com/bodywork-ml/bodywork-pymc3-project/blob/main/serve_model.py) for complete details),

```python
@app.post("/predict/v1.0.0/point", status_code=200)
def predict_point_estimate(request: PointPredictionRequest):
    """Return point-estimate prediction."""
    y_pred_samples = generate_label_samples(
        request.data.x, request.data.category, request.algo_param.n_samples
    )
    y_pred = np.median(y_pred_samples)
    return {"y_pred": y_pred, "algo_param": request.algo_param.n_samples}


@app.post("/predict/v1.0.0/interval", status_code=200)
def predict_interval(request: IntervalPredictionRequest):
    """Return point-estimate prediction."""
    y_pred_samples = generate_label_samples(
        request.data.x, request.data.category, request.algo_param.n_samples
    )
    y_hdi = pm.hdi(y_pred_samples, request.hdi_probability)
    return {
        "y_pred_lower": y_hdi[0],
        "y_pred_upper": y_hdi[1],
        "algo_param": request.algo_param.n_samples,
    }


@app.post("/predict/v1.0.0/density", status_code=200)
def predict_density(request: DensityPredictionRequest):
    """Return density prediction."""
    y_pred_samples = generate_label_samples(
        request.data.x, request.data.category, request.algo_param.n_samples
    )
    y_pred_density, bin_edges = np.histogram(
        y_pred_samples, bins=request.bins, density=True
    )
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {
        "y_pred_bin_mids": bin_mids.tolist(),
        "y_pred_density": y_pred_density.tolist(),
        "algo_param": request.algo_param.n_samples,
    }
```

Instances of data, serialised as JSON, can be sent to these endpoints as HTTP POST requests. The schema for the JSON data payload are defined by the classes that inherit from `pydantic.BaseModel`, as reproduced below.

```python
class FeatureDataInstance(BaseModel):
    """Pydantic schema for instances of feature data."""

    x: float
    category: int


class AlgoParam(BaseModel):
    """Pydantic schema for algorithm config."""

    n_samples: int = Field(N_PREDICTION_SAMPLES, gt=0)


class PointPredictionRequest(BaseModel):
    """Pydantic schema for point-estimate requests."""

    data: FeatureDataInstance
    algo_param: AlgoParam = AlgoParam()


class IntervalPredictionRequest(BaseModel):
    """Pydantic schema for interval requests."""

    data: FeatureDataInstance
    hdi_probability: float = Field(0.95, gt=0, lt=1)
    algo_param: AlgoParam = AlgoParam()


class DensityPredictionRequest(BaseModel):
    """Pydantic schema for density requests."""

    data: FeatureDataInstance
    bins: int = Field(5, gt=0)
    algo_param: AlgoParam = AlgoParam()

```

For more information on defining JSON schemas using Pydantic and FastAPI, see the [FastAPI docs](https://fastapi.tiangolo.com/tutorial/body/).

## Testing the web API Locally

TODO

## Configuring the Deployment

All configuration for Bodywork deployments must be kept in a [YAML](https://en.wikipedia.org/wiki/YAML) file, named `bodywork.yaml` and stored in the project’s root directory.  The `bodywork.yaml` required to deploy our web API is reproduced below.

```yaml
version: "1.0"
project:
  name: bodywork-pymc3-project
  docker_image: bodyworkml/bodywork-core:latest
  DAG: scoring-service
stages:
  scoring-service:
    executable_module_path: serve_model.py
    requirements:
      - arviz==0.11.2
      - boto3==1.17.60
      - fastapi==0.63.0
      - joblib==1.0.1
      - numpy==1.20.2
      - pymc3==3.11.2
      - uvicorn==0.13.4
    secrets:
      AWS_ACCESS_KEY_ID: aws-credentials
      AWS_SECRET_ACCESS_KEY: aws-credentials
      AWS_DEFAULT_REGION: aws-credentials
    cpu_request: 1.0
    memory_request_mb: 500
    service:
      max_startup_time_seconds: 60
      replicas: 1
      port: 8000
      ingress: true
logging:
  log_level: INFO
```

Bodywork will interpret this file as follows:

1. Start a Bodywork container on Kubernetes, to run a service stage called `scoring-service`.
2. Install the Python packages required to run `serve_model.py`.
3. Run `serve_model.py`.
4. Monitor  `scoring-service` and ensure that there is always at least one service replica available, at all times - i.e. it if fails for any reason, then immediately start another one.

Refer to the [Bodywork User Guide](https://bodywork.readthedocs.io/en/latest/user_guide/#user-guide) for a complete discussion of all the options available for deploying machine learning systems using Bodywork.

## Deploying the Prediction Service

The first thing we need to do, is to create and setup a Kubernetes [namespace](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/) for our deployment. A namespace can be thought of as a virtual cluster (within the cluster), where related resources can be grouped together. Use the Bodywork CLI to do this,

```text
bodywork setup-namespace pymc
```

TODO

```text
$ bodywork secret create \
    --namespace=pymc \
    --name=aws-credentials \
    --data AWS_ACCESS_KEY_ID=XX AWS_SECRET_ACCESS_KEY=XX AWS_DEFAULT_REGION=XX
```

The, execute the deployment using,

```text
$ bodywork deployment create \
    --namespace=pymc \
    --name=initial-deployment \
    --git-repo-url=https://github.com/bodywork-ml/bodywork-pymc3-project \
    --git-repo-branch=main
```

Use the kubectl to check if the deployment job has completed,

```text
$ kubectl -n pymc get jobs

NAME              COMPLETIONS   DURATION   AGE
initial-deployment   1/1           69s        2m
```

Now test the service is responding,

```text
$ curl http://YOU_CLUSTER_IP/pymc/bodywork-pymc3-project--scoring-service/predict/v1.0.0/point \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"data": {"x": 5, "category": 2}}'

{
  "y_pred_lower": 5.997068122059717,
  "y_pred_upper": 8.67981161246493,
  "algo_param": 100
}
```

Returning the same value we got when testing the service locally. Congratulations - you have just deployed a probabilistic program ready for production.
