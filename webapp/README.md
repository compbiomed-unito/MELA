# MELA web tool

MELA is a Streamlit decision-support tool that estimates a risk class for acute aortic syndrome in adults being evaluated for suspected disease. It combines votes from three trained models and displays the risk observed for the resulting class. The tool is for educational and research use only.

## Run locally

Use Python 3.12 and run these commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run mela.py
```

The app loads its input definitions from `interface_data.xlsx` and its preprocessing objects, thresholds, and trained models from `../models/`. Keep these files together when deploying it. The displayed percentages are risk estimates for the predicted class, not individualized probability estimates.
