#!/bin/bash

# Install the project in editable/developement mode
pip install -e .

# Run FastAPI app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
