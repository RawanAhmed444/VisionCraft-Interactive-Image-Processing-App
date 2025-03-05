Project Structure

This project is organized into the following structure:

pyQt
├── data                 # Dataset and input files
├── docs                 # Documentation files
├── notebooks            # Jupyter notebooks for testing and prototyping
├── resources            # Additional resources (e.g., images, configuration files)
├── src                  # Source code for the application
│   ├── classes          # Contains class definitions for the application
│   ├── functions        # Modular functions for specific tasks
│   ├── app.py           # Main application logic
│   ├── main.py          # Entry point of the application
│   ├── processor_factory.py # Factory pattern for selecting appropriate processors
│   ├── testing_processors.ipynb # Notebook for testing processing functions
│   ├── utils.py         # Utility functions used across the application
├── ui                   # User interface elements
├── .gitignore           # Specifies files and directories to ignore in version control
├── requirements.txt     # List of dependencies required by the project
└── setup.py             # Script for setting up the project environment

Key Components

data/: Stores raw data and datasets.

docs/: Contains project documentation and guides.

notebooks/: Jupyter notebooks for experimentation.

resources/: Additional assets such as images and config files.

src/: Core source code, including:

classes/: Python classes used in the app.

functions/: Reusable function modules.

app.py: Core logic of the PyQt application.

main.py: Main script to run the application.

processor_factory.py: Implements a factory pattern for dynamic processing.

testing_processors.ipynb: Testing notebook for processing components.

utils.py: Helper functions.

tests/: Test scripts to ensure code correctness.

ui/: UI components for the PyQt interface.

requirements.txt: Lists all dependencies.

