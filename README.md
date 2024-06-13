# Fake Job Detection System

## Project Overview

The Fake Job Detection System is a machine learning project designed to detect fraudulent job postings. By utilizing natural language processing (NLP) techniques and a Random Forest classifier, the system analyzes the textual content of job postings to distinguish between genuine and fake listings.

## Project Structure

- `Fake_job_detection_code.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and prediction.
- `fake_job_postings.csv`: Dataset containing job postings information.
- `README.md`: Project documentation.

## Key Features

- **Data Preprocessing**: Cleaning and preparing text data for analysis.
- **Feature Engineering**: Combining and transforming text data for model input.
- **Modeling**: Implementing a Random Forest classifier to detect fraudulent job listings.
- **Visualization**: Generating word clouds and bar charts to understand data distribution and model results.

## Technologies Used

- Python
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn
- Random Forest
- TF-IDF

## Data Description

The dataset contains the following features:

- `title`: The job position title.
- `location`: The job location.
- `department`: The department to which the job belongs.
- `salary_range`: The salary range for the job.
- `company_profile`: A description of the company.
- `description`: Detailed job description.
- `requirements`: Job requirements and qualifications.
- `benefits`: Additional benefits provided by the company.
- `telecommuting`: Indicates if telecommuting is allowed (1 for yes, 0 for no).
- `has_company_logo`: Indicates if the job posting has a company logo (1 for yes, 0 for no).
- `has_questions`: Indicates if the job posting has screening questions (1 for yes, 0 for no).
- `employment_type`: Type of employment (full-time, part-time, contract, etc.).
- `required_experience`: Level of experience required.
- `required_education`: Minimum education level required.
- `industry`: Industry sector of the job.
- `function`: Specific function or area of specialization.
- `fraudulent`: Indicates if the job posting is fraudulent (1 for yes, 0 for no).

## Installation and Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Fake-Job-Detection.git
    cd Fake-Job-Detection
    ```

2. **Install required dependencies**:
    Make sure you have Python and pip installed. Then, run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK resources**:
    In your Jupyter Notebook or Python script, ensure you download the necessary NLTK resources:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

4. **Run the Jupyter Notebook**:
    Open `Fake_job_detection_code.ipynb` in Jupyter Notebook and run the cells to preprocess the data, train the model, and make predictions.

## Results

The model achieved an accuracy of [insert accuracy here] in detecting fraudulent job postings. The confusion matrix and classification report provided detailed performance metrics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- The dataset used in this project was obtained from [source of the dataset].
- Thanks to the contributors of open-source libraries such as pandas, numpy, nltk, scikit-learn, matplotlib, and seaborn.

