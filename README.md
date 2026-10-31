# AI Fashion Stylist

## Project Overview
AI Fashion Stylist is a deep learning-based fashion recommendation application that helps users discover fashion items and check outfit compatibility.

The project combines image-based fashion search with outfit compatibility prediction to help users explore and create stylish outfits.

## Features
- **Fashion Search:** Find similar fashion items using image-based search with CLIP.
- **Outfit Compatibility:** Predict whether clothing items work well together using ResNet18.
- **Random Outfit Generator:** Generate outfit combinations from available fashion items.
- **Interactive Application:** Explore fashion recommendations through a Streamlit interface.

## Technologies Used
- Python
- PyTorch
- Torchvision
- ResNet18
- CLIP
- Streamlit
- NumPy
- Pandas

## Project Structure
- `app.py` – Main Streamlit application
- `AI_Fashion_Stylist.ipynb` – Colab notebook for model development and experiments
- `fashion_stylist_resnet18.pth` – Trained outfit compatibility model
- `fashion_search_data/` – Fashion search data and embeddings
- `images/` – Fashion item images
- `requirements.txt` – Project dependencies

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Keerthireddyyy/AI-Fashion-Stylist.git
cd AI-Fashion-Stylist
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```

Mac/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

## Deep Learning Models
- **ResNet18:** Used for outfit compatibility prediction.
- **CLIP:** Used for image-based fashion search and similarity.

## Project Goal
The goal of this project is to demonstrate how deep learning and image embeddings can be applied to fashion search and outfit recommendation.

## Author
<<<<<<< Updated upstream
Keerthi kalakata, Rishitha Gaddam, Sai Surya Siddhardha
=======
Keerthi kalakata, Rishitha Gaddam, Sai Surya Siddhardha Seera
>>>>>>> Stashed changes
