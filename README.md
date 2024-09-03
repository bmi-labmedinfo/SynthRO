<div id="top"></div>

<div align="center">
  <h1>
    SynthRO: a dashboard to evaluate and benchmark synthetic data
  </h1>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#extensibility">Extensibility</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project

The rapid increase in patient data collection by healthcare providers, governments, and private industries is generating vast and varied datasets that provide new insights into critical medical questions. Despite the rise of medical devices powered by Artificial Intelligence, research access to data remains restricted due to privacy concerns. One possible solution is to use Synthetic Data, which replicates the main statistical properties of real patient data. However, the lack of standardized evaluation metrics makes selecting appropriate synthetic data methods challenging. Effective evaluation must balance resemblance, utility, and privacy, but current benchmarking efforts are limited, necessitating further research.

To address this constraint, we've introduced SynthRO (Synthetic data Rank and Order), a user-friendly tool designed to benchmark synthetic health tabular data across various contexts. SynthRO provides accessible quality evaluation metrics and automated benchmarking, enabling users to identify the most suitable synthetic data models for specific applications by prioritizing metrics and delivering consistent quantitative scores.

<p align="right"><a href="#top">↰ Back To Top</a></p>

## Installation

This repository provides a Conda environment configuration file (`synthro_env.yml`) to streamline the setup process. Follow these steps to create the environment:

> [!IMPORTANT]
> Make sure you have Conda installed. If not, [install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) before proceeding.

### Steps to Create the Environment

1. **Create the Conda Environment**

    Run the following command to create the environment using the provided `.yml` file:

    ```bash
    conda env create -f synthro_env.yml
    ```

    This command will set up a Conda environment named according to specifications in the `synthro_env.yml` file.

2. **Activate the Environment**

    Once the environment is created, activate it using:

    ```bash
    conda activate synthro_env
    ```

### Running the Code

Once the virtual environment is activated, you can run the code using the following steps:

```bash
python SynthRO_app.py
```

### Additional Notes

To deactivate the environment, simply use:

```bash
conda deactivate
```

<p align="right"><a href="#top">↰ Back To Top</a></p>

> [!TIP]
> If you want to try the tool, [here](example%20datasets/) you will find an example of an original and synthetic dataset.


## Extensibility

The tool has a modular structure, allowing new sections and evaluation metrics to be added at any time. 

### Methodology

Regarding the methodological part, the code should be integrated into one of the classes already implemented in the `utils.py` script. For instance, if you want to add a new type of simulated attack among the privacy metrics, it should be added as a static method of the `Privacy` class:

```python
class Privacy:

    # Other implemented methods

    @staticmethod
    def new_simulated_attack():
        # Code for the new method
        pass
```

Afterwards, the new method must be invoked within the main script.

### Graphical Interface

The graphical interface was developed using the [Dash package](https://dash.plotly.com/) in Python. Once the new metric is defined, it can be integrated into the existing graphical elements or a new section can be created using the graphical elements provided by the package.

The `SynthRO_app.py` script is divided into well-defined sections, making it easy for the user to locate new graphical elements.

<p align="right"><a href="#top">↰ Back To Top</a></p>

## License

SynthRO © 2024 by Gabriele Santangelo is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), click for more information.

<p align="right"><a href="#top">↰ Back To Top</a></p>

