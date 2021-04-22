# Scikit-Learn, meet Production

> “*Deploying something useless into production, as soon as you can, is the right way to start a new project. It pulls unknown risk forward, opens up parallel streams of work, and establishes good habits.*”

This is a quote by [Pete Hodgson](https://blog.thepete.net/blog/2019/10/04/hello-production/), from his article ‘Hello, production’.  It in a nutshell, it explains the benefits of taking deployment pains early on in a software development project, and then using the initial deployment skeleton as the basis for rapidly delivering useful functionality into production.

The idea of making an initial ‘Hello, production’ release has had a big influence on how we think about the development of machine learning systems. We’ve mapped ‘Hello, Production’ into the machine learning space, as follows,

> *Train the simplest model conceivable and deploy it into production, as soon as you can.*

A reasonable ‘Hello, production’ model could be one that returns the most frequent class (for classification tasks), or the mean value (for regression tasks).  Scikit-Learn provides models for precisely this situation, in the `sklearn.dummy` sub-module. If the end-goal is to serve predictions via a web API, then the next step is to develop the server and deploy it into a production environment. Alternatively, if the model is going to be used as part of a batch job, then the next step is to develop the job and deploy that into production.

The advantage of following this process, is that is forces you to confront the following issues early on:

- Getting access to data.
- Getting access to (or creating) production environments.
- Defining the contract (or interface) with the consumers of the model’s output.
- Creating deployment pipelines (manual or automated), to deliver your application into production.

Each one of these issues is likely to involve input from people in other teams and is critical to overall success. Failure on any one of these can signal the end for a machine learning project, regardless of how well the models are performing. Success also demonstrates an ability to deliver functional software, which in our experience creates trust in a project, and often leads to more time being made available to experiment with training more complex model types.

Bodywork is laser-focused on making the deployment of machine learning projects, to Kubernetes, quick and easy. In what follows, we are going to show you how to use Bodywork to deploy a ‘Hello, production’ release for a hypothetical prediction service, using Scikit-Learn and FastAPI. We claim that it will take your under 15 minutes to work through the steps below, which includes setting-up a local Kubernetes cluster for testing.

## Step 0 - Before we get Started

Deploying machine learning projects using Bodywork requires you to have a [GitHub](https://github.com) account, Python 3.8 installed on your local machine and access to a [Kubernetes](https://en.wikipedia.org/wiki/Kubernetes) cluster. If you already have access to Kubernetes, then skip to Step 1, otherwise read-on to setup a single node Kubernetes cluster on your local machine, using Minikube.

### Minikube - Kubernetes for your Laptop

If you don’t have access to a Kubernetes cluster, then an easy way to get started is with [Minikube](https://minikube.sigs.k8s.io/docs/). If you are running on MacOS and with the [Homebrew](https://brew.sh) package manager available, then installing Minikube is as simple as running,

```shell
$ brew install minikube
```

If you’re running on Windows or Linux, then see the appropriate [installation instructions](https://minikube.sigs.k8s.io/docs/start/).

Once you have Minikube installed, start a cluster using the latest version of Kubernetes that Bodywork supports,

```shell
$ minikube start --kubernetes-version=v1.16.15
```

And then enable ingress, so we can route HTTP requests to services deployed using Bodywork.

```she’ll
$ minikube addons enable ingress
```

You’ll also need the cluster’s IP address, which you can get using,

```shell
$ minikube profile list
|----------|-----------|---------|--------------|------|----------|---------|-------|
| Profile  | VM Driver | Runtime |      IP      | Port | Version  | Status  | Nodes |
|----------|-----------|---------|--------------|------|----------|---------|-------|
| minikube | hyperkit  | docker  | 192.168.64.5 | 8443 | v1.16.15 | Running |     1 |
|----------|-----------|---------|--------------|------|----------|---------|-------|
```

When you’re done with this tutorial, the cluster can be powered-down using.

```shell
$ minikube stop
```

## Step 1 - Create a new GitHub Repository for the Project

Head over to GitHub and create a new public repository for this project - we called ours [bodywork-scikit-fastapi-project](https://github.com/bodywork-ml/bodywork-scikit-fastapi-project). If you want to use Bodywork with private repos, you’ll have to configure Bodywork to authenticate with GitHub via SSH. The [Bodywork User Guide](https://bodywork.readthedocs.io/en/latest/user_guide/#working-with-private-git-repositories-using-ssh) contains details on how to do this, but we recommend that you come back to this at a later date and continue with a public repository for now.

Next, clone your new repository locally,

```shell
$ git clone https://github.com/bodywork-ml/bodywork-scikit-fastapi-project.git
```

Create a dedicated Python 3.8 virtual environment in the root directory, and the activate it,

```shell
$ cd bodywork-scikit-fastapi-project
$ python3.8 -m venv .venv
$ source .venv/bin/activate
```

Finally, install the packages required for this project, as shown below,

```shell
$ pip install \
    bodywork==2.0.2 \
	scikit-learn==0.24.1 \
	numpy==1.20.2 \
	joblib==1.0.1 \
	fastapi==0.63.0 \
	uvicorn==0.13.4 
```

Then open-up an IDE to continue developing the service.

## Step 2 - Train a Dummy Model

We want to demonstrate a ‘Hello, production’ release, so we’ll train a Scikit-Learn `DummyRegressor`, configured to return the mean value of the labels in a training dataset, regardless of the feature data passed to it. This will still require you to acquire some data, one way or another.

For the purposes of this article, we have opted to create a synthetic one-dimensional regression dataset, where the only feature, `X`, has a 42% correlation with the labels, `y`, and both features and labels are [distributed normally](https://en.wikipedia.org/wiki/Normal_distribution). We have added this step to our training script, `train_model.py`, reproduced below. When you run the training script, it will train a `DummyRegressor` and save it in the project’s root directory as `dummy_model.joblib`.

Beyond use in ‘Hello, production’ releases, models such as this represent the most basic benchmark that any more sophisticated model type must out-perform - which is why the script also persists the model metrics in `dummy_model_metrics.txt`, for comparisons with future iterations.

```python
import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# create dummy regression data
n_observations = 1000
np.random.seed(42)
X = np.random.randn(n_observations)
y = 0.42 * X + np.sqrt(1 - 0.42 * 0.42) * np.random.randn(n_observations)

# train dummy model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train, y_train)

# compute dummy model metrics
mse = mean_squared_error(y_test, dummy_model.predict(X_test))

# persist dummy model and metrics
joblib.dump(dummy_model, 'dummy_model.joblib')
with open('dummy_model_metrics.txt', 'w') as f:
    f.write(f'mean_squared_error: {mse}\n')
```

## Step 3 - Develop a web API using FastAPI

The ultimate aim for our chosen machine learning system, is to serve predictions via a web API. Consequently, our initial ‘Hello, production’ release will need us to develop a skeleton web service that exposes the dummy model trained in Step 2. This is achieved in a Python module we’ve named `serve_model.py` , reproduced below, which you should also add to your project.

This module loads the trained model created in Step 2 and then configures [FastAPI](https://fastapi.tiangolo.com) to start a server with an HTTP endpoint at `/api/v1/`. Instances of data, serialised as JSON,  can be sent to this endpoint as HTTP POST requests. The schema for the JSON data payload is defined by the `FeatureDataInstanace` class, which for our example only expects a single `float` field named `X`. For more information on defining JSON schemas using Pydantic and FastAPI, see the [FastAPI docs](https://fastapi.tiangolo.com/tutorial/body/).

```python
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(debug=False)


class FeatureDataInstance(BaseModel):
    """Define JSON data schema for prediction requests."""
    X: float


@app.post('/api/v1/', status_code=200)
def predict(data: FeatureDataInstance):
    """Generate predictions for data sent to the /api/v1/ route."""
    prediction = model.predict([data.X])
    return {'y_pred': prediction[0]}


if __name__ == '__main__':
    model = joblib.load('dummy_model.joblib')
    uvicorn.run(app, host='0.0.0.0', workers=1)
```

Test the service locally by running `serve_model.py` ,

```shell
$ python serve_model.py
INFO:     Started server process [51987]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

And then in a new terminal, send the endpoint some data using `curl`,

```shell
$ curl http://localhost:8000/predict/v1/ \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": 42}'
{"y_pred":-0.0032494670211433195}
```

Which confirms that the service is working as expected.

## Step 4 - Configure Deployment

All configuration for Bodywork deployments must be kept in a [YAML](https://en.wikipedia.org/wiki/YAML) file, named `bodywork.yaml` and stored in the project’s root directory.  The `bodywork.yaml` required to deploy our ‘Hello, production’ release is reproduced below - add this file to your project.

```yaml
version: "1.0"
project:
  name: bodywork-scikit-fastapi-project
  docker_image: bodyworkml/bodywork-core:latest
  DAG: scoring-service
stages:
  scoring-service:
    executable_module_path: serve_model.py
    requirements:
      - fastapi==0.63.0
      - joblib==1.0.1
      - numpy==1.20.2
      - scikit-learn==0.24.1
      - uvicorn==0.13.4
    cpu_request: 0.5
    memory_request_mb: 100
    service:
      max_startup_time_seconds: 30
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

## Step 5 - Commit Files and Push to GitHub

The project is now ready to deploy, so the files must be committed and pushed to the remote repository we created on GitHub.

```shell
$ git add -A
$ git commit -m "Initial commit."
$ git push origin main
```

When triggered, Bodywork will clone the remote repository directly from GitHub, analyse the configuration in `bodywork.yaml` and then execute the deployment plan contained within it.

## Step 6 - Deploy to Kubernetes

The first thing we need to do, is to create and setup a Kubernetes [namespace](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/) for our deployment. A namespace can be thought of as a virtual cluster (within the cluster), where related resources can be grouped together. Use the Bodywork CLI to do this,

```shell
$ bodywork setup-namespace bodyworkml
```

The easiest way to run your first deployment, is to execute the Bodywork workflow-controller locally,

```shell
$ bodywork workflow \
    --namespace=bodyworkml \
    https://github.com/bodywork-ml/bodywork-scikit-fastapi-project.git \
    main
```

This will orchestrate deployment on your cluster and stream the logs to your terminal. Refer to the [Bodywork User guide](https://bodywork.readthedocs.io/en/latest/user_guide/#deploying-workflows)  to run the workflow-controller remotely.

## Step 7 - Test the Prediction API

Once the deployment has completed, the prediction service will be ready for testing. Bodywork will create ingress routes to your endpoint using the following scheme:

```md
/K8S_NAMESPACE/PROJECT_NAME--STAGE_NAME/
```

Such that we can make a request for a prediction using,

```shell
$ curl http://CLUSTER_IP/bodyworkml/bodywork-scikit-fastapi-project--scoring-service/api/v1/ \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": 42}'

{"y_pred": 0.0781994319124968}
```

Returning the same value we got when testing the service earlier on. Congratulations, you have just deployed your ‘Hello, production’ release!

## Where to go from here

If you used Minikube to test Bodywork locally, then the next logical step would be to deploy to a remote Kubernetes cluster. There are many options for creating managed Kubernetes clusters in the cloud - see [our recommendations](https://bodywork.readthedocs.io/en/latest/kubernetes/#managed-kubernetes-services)

If a web service isn’t a suitable ‘Hello, production’ release for your project, then check out the [Deployment Templates](https://bodywork.readthedocs.io/en/latest/template_projects/) for other project types that may be a better fit - e.g. batch jobs or Jupyter notebook pipelines.

When your ‘Hello, production’ release is operational and available within your organisation, it’s then time to start thinking about monitoring your service and collecting data to enable the training of the next iteration. Godspeed!

## Getting Help

If you run into any trouble, then don't hesitate to ask a question on our [discussion board](https://github.com/bodywork-ml/bodywork-core/discussions) and we'll step-in to help you out.
