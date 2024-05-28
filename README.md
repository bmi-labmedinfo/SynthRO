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

- To deactivate the environment, simply use:

    ```bash
    conda deactivate
    ```

- You can now work within this Conda environment to run the application.

<p align="right"><a href="#top">↰ Back To Top</a></p>

## License

SynthRO © 2024 by Gabriele Santangelo is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), click for more information.

<p align="right"><a href="#top">↰ Back To Top</a></p>

