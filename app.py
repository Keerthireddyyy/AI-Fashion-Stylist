import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="AI Fashion Stylist",
    page_icon="👗",
    layout="wide"
)

# ----------------------------------
# Dashboard Title
# ----------------------------------

st.title("👗 AI Fashion Stylist")

# ResNet18 Outfit Compatibility Model

class FashionResNet18(nn.Module):

    def __init__(self, resnet):
        super().__init__()

        self.resnet = resnet
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, images, mask):

        batch_size = images.size(0)
        num_items = images.size(1)

        images = images.view(
            batch_size * num_items,
            3,
            224,
            224
        )

        features = self.resnet(images)

        features = features.view(
            batch_size,
            num_items,
            512
        )

        mask = mask.unsqueeze(-1)
        features = features * mask

        outfit_features = features.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        output = self.classifier(outfit_features)

        return output.squeeze(1)

# Load trained outfit compatibility model

@st.cache_resource
def load_compatibility_model():

    # Create ResNet18 without downloading pretrained weights
    resnet = models.resnet18(weights=None)

    # Create our outfit compatibility model
    model = FashionResNet18(resnet)

    # Load the trained model weights
    model_path = "fashion_stylist_resnet18.pth"

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    return model


compatibility_model = load_compatibility_model()

st.success("Outfit compatibility model loaded successfully!")



# Calculate outfit compatibility

def score_outfit(image_paths):

    images = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = outfit_transform(image)
        images.append(image)

    outfit_tensor = torch.stack(images).unsqueeze(0)

    mask = torch.ones(
        1,
        len(images),
        dtype=torch.float32
    )

    with torch.no_grad():
        score = compatibility_model(outfit_tensor, mask)
        probability = torch.sigmoid(score).item()

    return probability


# Prepare images for outfit compatibility model

outfit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Calculate outfit compatibility

def score_outfit(image_paths):

    images = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = outfit_transform(image)
        images.append(image)

    outfit_tensor = torch.stack(images).unsqueeze(0)

    mask = torch.ones(
        1,
        len(images),
        dtype=torch.float32
    )

    with torch.no_grad():
        score = compatibility_model(outfit_tensor, mask)
        probability = torch.sigmoid(score).item()

    return probability

# ----------------------------------
# AI Fashion Search using CLIP
# ----------------------------------

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Load fashion item categories

@st.cache_data
def load_fashion_categories():

    categories = pd.read_csv(
        "fashion_search_data/fashion_categories.csv"
    )

    categories["item_id"] = categories["item_id"].astype(str)

    return categories


fashion_categories = load_fashion_categories()

st.success(
    f"Fashion categories loaded: {len(fashion_categories)} items"
)

# ----------------------------------
# Outfit Generator
# ----------------------------------

st.header("👗 Create Your Outfit")

outfit_type = st.selectbox(
    "Choose your outfit style:",
    [
        "Casual Outfit",
        "Dress Outfit",
        "Stylish Outfit"
    ]
)

outfit_templates = {
    "Casual Outfit": [
        "a shirt",
        "a pair of jeans",
        "a pair of shoes",
        "a handbag"
    ],

    "Dress Outfit": [
        "a dress",
        "a pair of shoes",
        "a handbag",
        "a pair of earrings"
    ],

    "Stylish Outfit": [
        "a shirt",
        "a pair of jeans",
        "a pair of shoes",
        "a handbag",
        "a pair of earrings"
    ]
}

selected_categories = outfit_templates[outfit_type]

st.write("### Your outfit includes:")

for category in selected_categories:
    st.write("•", category)



# Fashion Search - Complete Color-Based Outfit Recommendation

st.header("🔍 Search Fashion Items")

fashion_query = st.text_input(
    "What color or style outfit are you looking for?",
    placeholder="Example: red outfit, blue casual outfit, pink party outfit"
)

if fashion_query:

    # Load the CLIP model
    @st.cache_resource
    def load_clip_model():

        model_name = "openai/clip-vit-base-patch32"

        processor = CLIPProcessor.from_pretrained(model_name)

        model = CLIPModel.from_pretrained(model_name)

        model.eval()

        return model, processor

    # Load fashion images and their embeddings
    @st.cache_data
    def load_fashion_data():

        data = torch.load(
            "fashion_search_data/fashion_clip_embeddings.pt",
            map_location="cpu",
            weights_only=False
        )

        return data["image_embeddings"], data["item_ids"]

    # Load fashion categories
    @st.cache_data
    def load_fashion_categories():

        return pd.read_csv(
            "fashion_search_data/fashion_categories.csv"
        )

    model, processor = load_clip_model()

    image_embeddings, item_ids = load_fashion_data()

    fashion_categories = load_fashion_categories()

    # Convert item IDs to strings
    item_ids = [str(item_id) for item_id in item_ids]

    # Match item IDs with their categories
    category_lookup = dict(
        zip(
            fashion_categories["item_id"].astype(str),
            fashion_categories["category"]
        )
    )

    # Normalize image embeddings
    image_embeddings = image_embeddings.float()

    image_embeddings = image_embeddings / image_embeddings.norm(
        dim=-1,
        keepdim=True
    )

    # Outfit categories selected above
    selected_categories = outfit_templates[outfit_type]

    st.subheader(f"Complete Outfit for: {fashion_query}")

    st.write("Your outfit includes:")

    for category in selected_categories:
        st.write("•", category)

    result_columns = st.columns(len(selected_categories))

    # Search separately for each clothing category
    for position, category in enumerate(selected_categories):

        # Create a text prompt using the user's color/style and category
        category_prompt = f"{fashion_query}, {category}, fashion item"

        inputs = processor(
            text=[category_prompt],
            return_tensors="pt",
            padding=True
        )

        # Get text embedding
        with torch.no_grad():

            text_features = model.get_text_features(**inputs)

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True
            )

        # Calculate similarity with all fashion images
        similarities = (
            text_features @ image_embeddings.T
        ).squeeze(0)

        # Find images belonging to this category
        category_indices = [
            index
            for index, item_id in enumerate(item_ids)
            if category_lookup.get(item_id) == category
        ]

        with result_columns[position]:

            st.markdown(f"**{category}**")

            if category_indices:

                # Find the best matching item in this category
                category_scores = similarities[category_indices]

                best_position = torch.argmax(category_scores).item()

                best_index = category_indices[best_position]

                item_id = item_ids[best_index]

                image_path = f"images/{item_id}.jpg"

                if os.path.exists(image_path):

                    st.image(
                        image_path,
                        use_container_width=True
                    )

                    st.caption(f"Item ID: {item_id}")

                    st.caption(
                        f"Match: {similarities[best_index].item():.3f}"
                    )

                else:

                    st.warning("Fashion image not found.")

            else:

                st.warning(
                    f"No items found for category: {category}"
                )

st.write(
    "Deep Learning-Based Outfit Compatibility "
    "and Fashion Recommendation System"
)

st.divider()

# ----------------------------------
# Model Results
# ----------------------------------

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Custom CNN Validation",
    "50.0%"
)

col2.metric(
    "ResNet18 Validation",
    "62.5%"
)

col3.metric(
    "ResNet18 Test Accuracy",
    "65.5%"
)

# ----------------------------------
# Training Loss
# ----------------------------------

st.header("📉 ResNet18 Training Progress")

epochs = [1, 2, 3, 4, 5, 6]

losses = [
    0.6951,
    0.6549,
    0.6238,
    0.5967,
    0.5556,
    0.5246
]

training_data = pd.DataFrame({
    "Epoch": epochs,
    "Training Loss": losses
})

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    epochs,
    losses,
    marker="o"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss Across 6 Epochs")
ax.grid(True)

st.pyplot(fig)

st.dataframe(
    training_data,
    use_container_width=True
)

# ----------------------------------
# Confusion Matrix
# ----------------------------------

st.header("🎯 Test Confusion Matrix")

cm = [
    [131, 69],
    [69, 131]
]

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_xticklabels([
    "Incompatible",
    "Compatible"
])

ax.set_yticks([0, 1])
ax.set_yticklabels([
    "Incompatible",
    "Compatible"
])

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center",
            fontsize=14
        )

st.pyplot(fig)

# ----------------------------------
# Recommendations
# ----------------------------------

st.header("👚 Top 5 Fashion Recommendations")

recommendations = pd.DataFrame({
    "Item ID": [
        "41533392",
        "183863886",
        "174610996",
        "193587212",
        "112022249"
    ],

    "Compatibility Score (%)": [
        86.31,
        85.07,
        80.11,
        78.52,
        28.41
    ]
})

st.dataframe(
    recommendations,
    use_container_width=True
)
# Display fashion recommendation images

st.subheader("👗 Recommended Fashion Items")

image_columns = st.columns(5)

for index, row in recommendations.iterrows():
    item_id = str(row["Item ID"])
    score = row["Compatibility Score (%)"]

    image_path = f"images/{item_id}.jpg"

    with image_columns[index]:
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
            st.write(f"**Item ID:** {item_id}")
            st.write(f"**Compatibility:** {score}%")
        else:
            st.warning(f"Image not found: {item_id}")

fig, ax = plt.subplots(figsize=(8, 4))

ax.barh(
    recommendations["Item ID"],
    recommendations["Compatibility Score (%)"]
)

ax.set_xlabel("Compatibility Score (%)")
ax.set_title("Top 5 Recommended Fashion Items")
ax.set_xlim(0, 100)

st.pyplot(fig)

# ----------------------------------
# Best Recommendation
# ----------------------------------

st.success(
    "Best Recommendation: Item 41533392 "
    "| Compatibility Score: 86.31%"
)

st.divider()

st.caption(
    "AI Fashion Stylist | Polyvore Dataset | ResNet18"
)
