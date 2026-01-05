FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "energy_optimizer", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Set environment name
ENV CONDA_DEFAULT_ENV=energy_optimizer

# Default command: start Jupyter notebook
CMD ["conda", "run", "--no-capture-output", "-n", "energy_optimizer", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
