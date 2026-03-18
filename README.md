# 🐾 Project Marley

This is a group capstone project completed for the M.S. Data Science & Analytics program at San Jose State University.

<p>
This project was done by a group of dog-lovers who wanted to create an LLM-powered chatbot that answers dog health and behavior questions, to provide early answers to dog owners before reaching out to a veterinarian. 
</p>

*Note:* This project experimented with a few other LLMs (such as Falcon-7B and more), but this repository only contains code used to experiment using the FLAN model.


## 📁 Project Structure

```
projectmarley/
├── 01_data/          # Raw and processed datasets
├── 02_models/        # Trained models and training notebooks
├── 03_gradio/        # Gradio app for interactive inference
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/dahlia25/projectmarley.git
   cd projectmarley
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > If no `requirements.txt` is present, install common dependencies manually:
   > ```bash
   > pip install gradio pandas numpy scikit-learn jupyter
   > ```

## 📊 Data (`01_data/`)

This directory contains the datasets used for training and evaluation.

- Raw data files are stored here for reproducibility.
- Any preprocessing steps are documented in the notebooks within `02_models/`.

## 🤖 Models (`02_models/`)

This directory contains Jupyter Notebooks and scripts for:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Saved model artifacts (`.pkl`, `.h5`, etc.)

To run the notebooks:

```bash
jupyter notebook 02_models/
```

## 🖥️ Gradio App (`03_gradio/`)

An interactive web demo powered by [Gradio](https://gradio.app/) that lets you run inference with the trained model directly in your browser.

### Running the App

```bash
cd 03_gradio
python app.py
```

Then open your browser and navigate to `http://localhost:7860`.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Jupyter Notebook | Data exploration & model development |
| Gradio | Interactive web UI for model demos |
| pandas / NumPy | Data manipulation |
| scikit-learn | Machine learning |

## 🤝 Acknowledgement

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is open source. Please add a license file if you intend to share or distribute this work.