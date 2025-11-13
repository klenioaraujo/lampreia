#!/bin/bash

# ΨQRH Lampreia XeLaTeX Compilation Script
# This script compiles the XeLaTeX paper with proper error handling

set -e  # Exit on any error

PAPER_NAME="psi_qrh_lampreia_paper"
TEX_FILE="${PAPER_NAME}.tex"
PDF_FILE="${PAPER_NAME}.pdf"

echo "=========================================="
echo "ΨQRH Lampreia XeLaTeX Paper Compilation"
echo "=========================================="

# Check if XeLaTeX is installed
USE_DOCKER=false
if ! command -v xelatex &> /dev/null; then
    echo "⚠️  XeLaTeX not found locally!"

    # Check if Docker is available
    if command -v docker &> /dev/null; then
        echo "✅ Docker found - will use Docker for compilation"
        USE_DOCKER=true
    else
        echo "❌ Docker not found either!"
        echo ""
        echo "Installation options:"
        echo ""
        echo "Option 1 - Install XeLaTeX locally:"
        echo "Ubuntu/Debian:"
        echo "  sudo apt update"
        echo "  sudo apt install texlive-xetex texlive-latex-extra texlive-fonts-recommended"
        echo "  sudo apt install texlive-bibtex-extra biber"
        echo ""
        echo "macOS (with Homebrew):"
        echo "  brew install mactex"
        echo ""
        echo "Arch Linux:"
        echo "  sudo pacman -S texlive-core texlive-bin biber"
        echo ""
        echo "Option 2 - Install Docker:"
        echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "  sudo sh get-docker.sh"
        echo ""
        exit 1
    fi
else
    echo "✅ XeLaTeX found locally"
    echo "✅ XeLaTeX version: $(xelatex --version | head -n 1)"
fi

# Check if required files exist
if [ ! -f "$TEX_FILE" ]; then
    echo "❌ Main TeX file not found: $TEX_FILE"
    exit 1
fi

if [ ! -f "references.bib" ]; then
    echo "❌ Bibliography file not found: references.bib"
    exit 1
fi

if [ ! -f "lampreia.png" ]; then
    echo "⚠️  Logo file not found: lampreia.png (paper will compile without logo)"
fi

echo ""
echo "📄 Compiling XeLaTeX document..."
echo "   This may take a few minutes..."
echo ""

if [ "$USE_DOCKER" = true ]; then
    echo "🐳 Using Docker for compilation..."

    # Build Docker image if needed
    if ! docker images | grep -q "psiqrh-lampreia-paper"; then
        echo "Building Docker image..."
        docker build -t psiqrh-lampreia-paper . || {
            echo "❌ Docker build failed"
            exit 1
        }
    fi

    # First compilation
    echo "Step 1/3: Initial XeLaTeX compilation..."
    if docker run --rm -v "$(pwd)":/workspace psiqrh-lampreia-paper \
        xelatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1; then
        echo "✅ Initial compilation successful"
    else
        echo "❌ Initial compilation failed"
        echo "Check the log file: ${PAPER_NAME}.log"
        exit 1
    fi
else
    # First compilation
    echo "Step 1/3: Initial XeLaTeX compilation..."
    if xelatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1; then
        echo "✅ Initial compilation successful"
    else
        echo "❌ Initial compilation failed"
        echo "Check the log file: ${PAPER_NAME}.log"
        exit 1
    fi
fi

# Bibliography processing
echo "Step 2/3: Processing bibliography..."
if [ "$USE_DOCKER" = true ]; then
    # Use Docker for bibliography processing
    if docker run --rm -v "$(pwd)":/workspace psiqrh-lampreia-paper \
        biber "$PAPER_NAME" > /dev/null 2>&1; then
        echo "✅ Bibliography processed with Biber (Docker)"
    else
        echo "⚠️  Biber failed, trying BibTeX..."
        if docker run --rm -v "$(pwd)":/workspace psiqrh-lampreia-paper \
            bibtex "$PAPER_NAME" > /dev/null 2>&1; then
            echo "✅ Bibliography processed with BibTeX (Docker)"
        else
            echo "❌ Bibliography processing failed"
        fi
    fi
else
    # Use local tools
    if command -v biber &> /dev/null; then
        if biber "$PAPER_NAME" > /dev/null 2>&1; then
            echo "✅ Bibliography processed with Biber"
        else
            echo "⚠️  Biber failed, trying BibTeX..."
            if bibtex "$PAPER_NAME" > /dev/null 2>&1; then
                echo "✅ Bibliography processed with BibTeX"
            else
                echo "❌ Bibliography processing failed"
            fi
        fi
    else
        echo "⚠️  Biber not found, trying BibTeX..."
        if bibtex "$PAPER_NAME" > /dev/null 2>&1; then
            echo "✅ Bibliography processed with BibTeX"
        else
            echo "❌ Bibliography processing failed"
        fi
    fi
fi

# Final compilations
echo "Step 3/3: Final XeLaTeX compilations..."
for i in {1..2}; do
    if [ "$USE_DOCKER" = true ]; then
        if docker run --rm -v "$(pwd)":/workspace psiqrh-lampreia-paper \
            xelatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1; then
            echo "✅ Final compilation $i successful (Docker)"
        else
            echo "❌ Final compilation $i failed"
            echo "Check the log file: ${PAPER_NAME}.log"
            exit 1
        fi
    else
        if xelatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1; then
            echo "✅ Final compilation $i successful"
        else
            echo "❌ Final compilation $i failed"
            echo "Check the log file: ${PAPER_NAME}.log"
            exit 1
        fi
    fi
done

echo ""
echo "=========================================="
echo "🎉 COMPILATION COMPLETE!"
echo "=========================================="
echo ""
echo "📄 Generated files:"
echo "   • ${PDF_FILE} (main document)"
echo "   • ${PAPER_NAME}.log (compilation log)"
echo "   • ${PAPER_NAME}.aux (auxiliary file)"
echo "   • ${PAPER_NAME}.bbl (bibliography)"
echo ""
echo "📖 To view the PDF:"
echo "   • Linux: xdg-open ${PDF_FILE}"
echo "   • macOS: open ${PDF_FILE}"
echo "   • Windows: start ${PDF_FILE}"
echo ""
echo "🧹 To clean auxiliary files:"
echo "   make clean    # or rm *.aux *.log *.bbl *.blg"
echo ""
echo "=========================================="