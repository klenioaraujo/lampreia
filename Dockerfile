# XeLaTeX Compilation Environment for ΨQRH Lampreia Paper
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    texlive-xetex \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-bibtex-extra \
    biber \
    python3 \
    python3-pip \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for any preprocessing
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    scipy

# Set working directory
WORKDIR /workspace

# Copy paper files
COPY psi_qrh_lampreia_paper.tex /workspace/
COPY references.bib /workspace/
COPY lampreia.png /workspace/

# Default command
CMD ["bash"]