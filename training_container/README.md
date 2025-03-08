Certainly! Here's the README written in Markdown:

```markdown
# Stock Price Prediction Docker Container

## Overview

This Docker container serves as a comprehensive solution for training and predicting stock prices using machine learning models. It includes both model training and prediction code, along with the necessary stock data. The container is designed for use by the University of Virginia's College at Wise - Cyber Security & Robotics Club for stock price prediction tasks.

## Docker Container Details

### Requisites

- **Local Setup:** Ensure that you have Docker installed on your machine.
- **Docker Setup:** The Docker container is configured to facilitate both model training and stock price prediction. Place your code files and stock data in the 'FilesToUpload' directory, which will be automatically uploaded to the '/app' directory in the container during build and run processes.

### Designed By
Jason Mensah-Homiah  
LinkedIn: [JasonMH22345](https://www.linkedin.com/in/jasonmh22345/)  
Date: 01/21/2024

### Software Availability

The Docker container includes the following pre-installed Python libraries:
- pandas
- torch
- scikit-learn
- numpy
- flask

These libraries are utilized for data manipulation, machine learning, and creating the REST API for stock price prediction.

### SSH Access

SSH access is not enabled in this container as it is designed for running code related to stock price prediction.

### Default Terminal

The default terminal for the container is `/bin/bash`.

## Usage

### Build Docker Image
```bash
docker build -t stock_prediction_image .
```

### Run Docker Container (Generic)
```bash
docker run -it -p 5000:5000 stock_prediction_image
```

This command runs the container and maps port 5000 for accessing the Flask app.

### Run Docker Container (Customized)
```bash
docker run -it -p 5000:5000 --gpus all --memory=80g --memory-swap=80g --shm-size=8g stock_prediction_image
```

Options breakdown:
- `-it`: Enables an interactive terminal session.
- `-p 5000:5000`: Maps port 5000 for accessing the Flask app.
- `--gpus all`: Allocates all available GPUs to the container.
- `--memory=8g --memory-swap=8g --shm-size=4g`: Allocates specific memory resources.

## Container Details

- OS: Ubuntu 22.04.2 LTS
- Python Version: 3.8.12

## Additional Libraries

The container includes pre-installed libraries such as pandas, torch, scikit-learn, numpy, and flask. These libraries are essential for data processing, machine learning model development, and creating the REST API.

## Security Considerations

Use this Docker container in a secure environment, especially when dealing with sensitive financial data. Ensure that the container is appropriately configured to prevent potential vulnerabilities. Perform regular security audits and updates to keep the container secure.
```

This Markdown document provides a formatted README with the adjusted run command to give the container access to all GPUs.
