#!/usr/bin/env python3
"""
Startup script for the Hate Speech Detection API
This script handles model initialization and training if needed
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and log the result"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Main startup function"""
    logger.info("Starting Hate Speech Detection API setup...")
    
    # Check if we're in the right directory
    if not os.path.exists('model.py'):
        logger.error("model.py not found. Please run this script from the HateSpeech directory.")
        sys.exit(1)
    
    # Check if dataset exists, if not create it
    if not os.path.exists('enhanced_dataset.csv'):
        logger.info("Dataset not found. Creating enhanced dataset...")
        if not run_command('python data_generator.py', "Creating enhanced dataset"):
            logger.warning("Failed to create dataset. API will start without training data.")
    
    # Check if model exists, if not train it
    if not os.path.exists('hate_speech_detector_lstm.h5'):
        logger.info("Trained model not found. Training model...")
        if not run_command('python train.py', "Training model"):
            logger.warning("Failed to train model. API will start without pre-trained model.")
            logger.info("You can train the model later using the /train endpoint.")
    else:
        logger.info("Pre-trained model found.")
    
    logger.info("Setup complete. Starting Flask application...")
    
    # Start the Flask app
    os.system('python app.py')

if __name__ == '__main__':
    main() 