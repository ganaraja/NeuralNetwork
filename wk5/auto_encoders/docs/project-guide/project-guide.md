# Steps to execute this project:

1. Open the auto_encoders project in your IDE.
2. Enter the auto_encoders folder in your terminal, and run uv sync
3. Activate your environment using source .venv/bin/activate
4. Create and update the .env file.
5. Download the trees dataset provided on the course portal if you have not already.
6. Download any 5-6 random images and save them in another folder (we have used images of our team members for this class).
7. In the `config.yaml` file of your project, update the following:
    - `data` directory paths for MNIST and trees dataset. (If you have downloaded the MNIST dataset in the previous lab sessions already, you can use the same directory)
    - `results` path for the trees dataset.
    - `data` path for the folder with the random images.
8. Ctrl+S to save changes to the `config.yaml` file.
9. Run the file at `src/svlearn_autoencoders/scripts/resnet50_auto_encoder_main.py` to train the model. This is for Notebook 02-03.
10. You can now run the notebooks 01-03 at `docs/notebooks`.

## Homework

11. For notebooks 04-05 (homework), you have to train the model prior to running the notebooks, by running the training files at `src/svlearn_autoencoders/scripts/resnet50_denoising_auto_encoder_main.py` and `src/svlearn_autoencoders/scripts/resnet50_masked_auto_encoder_main.py`.