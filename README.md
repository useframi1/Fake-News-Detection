# Fake-News-Detection

Fake News Detection Using Deep Learning

## How to Run

### Setup

1. **Download the dataset**:

    - Download and unzip the dataset from [this Google Drive link](https://drive.google.com/drive/folders/1wXKx4VZpOuVcnA3jPwuDikXNM1_2-jPs?usp=drive_link)
    - Extract and place the contents in a folder called `data` in the project root

2. **Download the model weights**:

    - Download the pre-trained model weights from [this Google Drive link](https://drive.google.com/drive/folders/1X0MYdQNpRe5uMX5liKnw7GOaERQQqPfF?usp=drive_link)
    - Place the weight files in a folder called `models` in the project root

3. **Install the required dependencies**:
    - `pip install -r requirements.txt`

### Running the Detection System

To run the fake news classification pipeline, simply execute:

```bash
python main.py
```

This will load the data, train the model, and evaluate the performance. It will also run SHAP explainer to explain the model's predictions.
