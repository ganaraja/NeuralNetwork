# Steps to execute this project

1. Open the cnn_classification project in your IDE.
2. Enter the cnn_classification folder in your terminal, and run `uv sync`
3. Activate your environment using `source .venv/bin/activate`
4. Create and update the .env file.
5. Download the "trees" dataset provided, unzip and store it in your desired directory.
6. In the `config.yaml` file of your project, update the `data` folder path for the dataset accordingly.
7. Create an empty folder, say "trees" for your model to be saved, preferably within your results directory.
8. In the `config.yaml` file of your project, update the `results` folder path to the newly created folder accordingly.
9. Ctrl+S to save changes to the `config.yaml` file.
10. Download any random image from the internet of an oak tree and willow tree. This will be used while running the code.
11. To run **Tree classification**: Run the jupyter notebook file at `docs/notebooks/CNN_trees_classification.ipynb`