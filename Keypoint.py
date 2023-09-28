import streamlit as st
import os
import cv2
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
# Configuration Dictionary
config = {
    'CSV': {
        'COLUMNS_TO_READ': ['ImageFile'],
    },
    'Paths': {
        'OLD_BASE_PATH': "D:\\DOE\\",
        'DATA_FILE': "combined_output_cleaned.csv",
        'NEW_BASE_PATH': "C:\\Users\\aaron.woods\\OneDrive - Thermo Fisher Scientific\\Desktop\\Dec 24\\",
    },
    'SAMPLE_FRAC': 1,
}

def get_file_path(base_path: str, filename: str) -> str:
    return os.path.join(base_path, filename)

def read_csv(config: Dict) -> pd.DataFrame:
    data_file_path = get_file_path(config['Paths']['NEW_BASE_PATH'], config['Paths']['DATA_FILE'])
    if not os.path.exists(data_file_path):
        logger.error(f"File does not exist: {data_file_path}")
        return pd.DataFrame()

    try:
        data = pd.read_csv(data_file_path, usecols=config['CSV']['COLUMNS_TO_READ'])
        sample_frac = config.get('SAMPLE_FRAC', 1.0)
        if 0 < sample_frac < 1.0:
            data = data.sample(frac=sample_frac).reset_index(drop=True)
        logger.info("Data read successfully.")
    except Exception as e:
        logger.error(f"Could not read data - {e}")
        return pd.DataFrame()
    return data

def update_image_paths(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df['ImageFile'] = df['ImageFile'].str.replace(config['Paths']['OLD_BASE_PATH'], config['Paths']['NEW_BASE_PATH'], regex=False)
    logger.info("Image paths updated.")
    return df

def display_image_with_annotation(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder for image display
    image_display = st.empty()
    
    # Session state for image index and annotation
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []

    # Show image
    image_path = df.iloc[st.session_state.image_index]['ImageFile']
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_display.image(image, caption=image_path, use_column_width=True, channels='RGB')

    # Annotate image
    # col1, col2 = st.beta_columns(2) Module "Streamlit" has no attribute "beta_columns"
    col1, col2 = st.columns(2)
    with col1:
        x_coord = st.number_input(f"Keypoint X-coordinate for {image_path}", value=0)
    with col2:
        y_coord = st.number_input(f"Keypoint Y-coordinate for {image_path}", value=0)
    if st.button("Annotate"):
        st.session_state.annotations.append((x_coord, y_coord))
        if st.session_state.image_index < len(df) - 1:
            st.session_state.image_index += 1
        else:
            st.success("All images annotated!")
            df["Keypoint_X"] = [coord[0] for coord in st.session_state.annotations]
            df["Keypoint_Y"] = [coord[1] for coord in st.session_state.annotations]
            st.session_state.image_index = 0  # Reset for next run
            st.session_state.annotations = []

    return df

if __name__ == "__main__":
    st.title("Interactive Image Keypoint Annotation")
    
    data = read_csv(config)
    if not data.empty:
        data = update_image_paths(data, config)
        annotated_data = display_image_with_annotation(data)
        
        if st.button("Save Annotations to CSV"):
            annotated_data.to_csv(get_file_path(config['Paths']['NEW_BASE_PATH'], "annotated_data.csv"), index=False)
            st.success("Annotated data saved successfully!")