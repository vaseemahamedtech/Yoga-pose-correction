"""
Dataset Downloader and Preparation Script
Downloads yoga pose datasets from Kaggle and other sources
"""

import os
import sys

def print_download_instructions():
    """Print comprehensive dataset download instructions"""
    instructions = """
    ╔════════════════════════════════════════════════════════════════╗
    ║           YOGA POSE DATASET DOWNLOAD INSTRUCTIONS              ║
    ╚════════════════════════════════════════════════════════════════╝

    📥 STEP-BY-STEP GUIDE:

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OPTION 1: Yoga-82 Dataset (RECOMMENDED - 82 poses, 28K+ images)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Visit: https://sites.google.com/view/yoga-82/home
    2. Click "Dataset Download" 
    3. Fill the Google Form with your details
    4. You'll receive a download link via email
    5. Download the zip file
    6. Extract to: YogaPoseML/dataset/raw/yoga82/

    Expected structure:
    dataset/raw/yoga82/
        ├── train/
        │   ├── pose1/
        │   ├── pose2/
        │   └── ...
        └── test/
            ├── pose1/
            ├── pose2/
            └── ...

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OPTION 2: Yoga-16 Dataset (16 poses, good for quick testing)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Install Kaggle API:
       pip install kaggle

    2. Setup Kaggle credentials:
       - Go to: https://www.kaggle.com/settings
       - Click "Create New API Token"
       - Save kaggle.json to ~/.kaggle/
       - chmod 600 ~/.kaggle/kaggle.json

    3. Download dataset:
       kaggle datasets download -d mohiuddin2531/yoga-16
       unzip yoga-16.zip -d YogaPoseML/dataset/raw/yoga16/

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OPTION 3: Yoga Poses Dataset (5 poses, lightweight)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Visit Kaggle: https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification
    2. Download dataset
    3. Extract to: YogaPoseML/dataset/raw/yoga5/

    OR using Kaggle API:
       kaggle datasets download -d ujjwalchowdhury/yoga-pose-classification
       unzip yoga-pose-classification.zip -d YogaPoseML/dataset/raw/yoga5/

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OPTION 4: Create Your Own Dataset
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Create folders for each pose:
       dataset/raw/custom/
           ├── mountain/
           ├── warrior2/
           ├── tree/
           ├── downward_dog/
           └── ... (more poses)

    2. Add images (jpg, png) to each folder
    3. Minimum 50-100 images per pose recommended

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    NEXT STEPS AFTER DOWNLOAD:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Preprocess the dataset:
       python backend/data_preprocessor.py

    2. Train ML models:
       python backend/ml_trainer.py

    3. Train DL models:
       python backend/dl_trainer.py

    4. Start the application:
       cd backend && uvicorn main:app --reload

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ADDITIONAL DATASETS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    • Yoga Posture Dataset (3K images):
      https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset

    • Yoga-for-all (11K images, 10 poses):
      https://data.mendeley.com/datasets/jc4mmnvcdk/1

    • Yoga Pose Dataset (Figshare, 10 poses):
      https://figshare.com/articles/dataset/Yoga_Pose_Dataset/15112320

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ⚠️  NOTE: After downloading any dataset, make sure to:
    1. Check the folder structure
    2. Verify images are readable
    3. Run the preprocessor to extract features

    📧 Need help? Check the README.md file for more details.

    ═══════════════════════════════════════════════════════════════════
    """
    print(instructions)

if __name__ == "__main__":
    print_download_instructions()

    # Check if dataset exists
    dataset_paths = [
        'dataset/raw/yoga82',
        'dataset/raw/yoga16',
        'dataset/raw/yoga5',
        'dataset/raw/custom'
    ]

    found_datasets = [path for path in dataset_paths if os.path.exists(path)]

    if found_datasets:
        print(f"\n✅ Found datasets in:")
        for path in found_datasets:
            file_count = sum(len(files) for _, _, files in os.walk(path))
            print(f"   - {path} ({file_count} files)")
        print("\n🚀 You can now run: python backend/data_preprocessor.py")
    else:
        print("\n⚠️  No datasets found. Please follow the instructions above.")
