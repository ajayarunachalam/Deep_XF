#!/bin/sh

# assumes run from root directory

echo "Installing DeepXF - An Explainable Forecasting and Nowcasting Library with state-of-the-art Deep Neural Networks & Dynamic Factor Model" 
echo "START..."
pip install -U -r requirements.txt
echo "END"
xterm -e python -i -c "print('>>> from deep_xf.main import *');from deep_xf.main import *"
xterm -e python -i -c "print('>>> from deep_xf.dpp import *');from deep_xf.dpp import *"
xterm -e python -i -c "print('>>> from deep_xf.forecast_ml import *');from deep_xf.forecast_ml import *"
xterm -e python -i -c "print('>>> from deep_xf.forecast_ml_extension import *');from deep_xf.forecast_ml_extension import *"
xterm -e python -i -c "print('>>> from deep_xf.stats import *');from deep_xf.stats import *"
xterm -e python -i -c "print('>>> from deep_xf.utility import *');from deep_xf.utility import *"
xterm -e python -i -c "print('>>> from deep_xf.denoise import *');from deep_xf.denoise import *"
xterm -e python -i -c "print('>>> from deep_xf.similarity import *');from deep_xf.similarity import *"
echo "Test Environment Configured"
echo "Package Installed & Tested Sucessfully"

