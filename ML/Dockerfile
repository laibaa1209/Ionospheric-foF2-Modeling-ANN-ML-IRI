# Use official TensorFlow GPU image (already includes CUDA + cuDNN)
FROM tensorflow/tensorflow:2.15.0-gpu

# Install optional useful packages
RUN apt-get update && apt-get install -y \
    git nano && \
    pip install --upgrade pip

# Install your dependencies here
# Example:
# RUN pip install numpy pandas matplotlib scikit-learn jupyter notebook
RUN pip install numpy pandas matplotlib

# Set working directory
WORKDIR /workspace

# Copy project (optional)
COPY . /workspace

CMD [ "python3", "ann.py" ]
