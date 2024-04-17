## END to END Machine Learning Project For Student preformance 
This project aims to predict student performance based on various factors such as gender, ethnicity, parental education level, lunch type, test preparation course, reading score, and writing score.
![Screenshot 2024-04-18 at 12 59 21 AM](https://github.com/Equinox-M/Student_Preformance_ML/assets/92346639/a129bd51-213b-41e0-9f37-277fb127dc70)

## Project Structure

- **.ebextensions**: Contains configuration files for AWS Elastic Beanstalk deployment.
  - `python.config`: Configuration file for setting up Python environment on AWS Elastic Beanstalk.

- **artifacts**: Directory containing artifacts generated during model training.
  - `model.pkl`: Serialized trained machine learning model.
  - `preprocessor.pkl`: Serialized data preprocessor object.

- **data.csv**: Raw dataset containing student performance data.
- **model_trainer.py**: Module for training machine learning models.
- **src**: Source code directory.
  - **components**: Contains modules for data ingestion, transformation, and model training.
    - `data_ingestion.py`: Module for data ingestion from CSV files.
    - `data_transformation.py`: Module for data preprocessing and transformation.
    - `model_trainer.py`: Module for training machine learning models.
  - **pipeline**: Contains utility modules for logging, exceptions, and data preprocessing.
    - `exception.py`: Custom exception classes.
    - `logger.py`: Logging utility.
    - `utils.py`: Utility functions.
  - **templates**: HTML templates for the web application.
    - `home.html`: Home page template.
    - `index.html`: Index page template.
- **.gitignore**: Git ignore file to exclude certain files and directories from version control.
- **README.md**: This file.
- **app.py**: Flask application script.
- **requirements.txt**: List of Python dependencies required for the project.
- **setup.py**: Script for packaging the project for distribution.

## Usage

1. Install the required Python dependencies using `pip install -r requirements.txt`.
2. Run the Flask application using `python app.py`.
3. Access the application at `http://localhost:5000` in your web browser.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.
