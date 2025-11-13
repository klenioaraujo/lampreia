# ΨQRH Lampreia: XeLaTeX Technical Paper

## Overview

This directory contains the XeLaTeX source files for the comprehensive technical paper on **ΨQRH Lampreia**, a multi-teacher semantic knowledge distillation framework based on the Quaternionic Recursive Harmonic Wavefunction (ΨQRH) architecture.

## Files Structure

```
lampreia/
├── psi_qrh_lampreia_paper.tex    # Main XeLaTeX document
├── references.bib                # Bibliography database
├── Makefile                      # Compilation automation
├── compile_paper.sh             # Interactive compilation script
├── PAPER_README.md              # This file
├── lampreia.png                 # Logo for title page
├── psi_qrh_benchmark_lampreia.py # Implementation code
└── README.md                    # GitHub README
```

## Compilation Requirements

### System Requirements

- **XeLaTeX**: Modern LaTeX distribution with Unicode support
- **Fonts**: TeX Gyre fonts (included in most distributions)
- **Bibliography**: Biber or BibTeX for reference processing

### Installation (Ubuntu/Debian)

```bash
# Install TeX Live with XeLaTeX
sudo apt update
sudo apt install texlive-xetex texlive-latex-extra
sudo apt install texlive-fonts-recommended texlive-bibtex-extra
sudo apt install biber

# Optional: Install additional fonts
sudo apt install fonts-lmodern fonts-texgyre
```

### Installation (macOS with Homebrew)

```bash
# Install MacTeX (includes XeLaTeX)
brew install mactex

# Or install BasicTeX + additional packages
brew install basictex
sudo tlmgr install xetex biblatex biber
```

### Installation (Arch Linux)

```bash
sudo pacman -S texlive-core texlive-bin biber
```

## Compilation Methods

### Method 1: Interactive Script (Recommended)

```bash
# Make executable and run
chmod +x compile_paper.sh
./compile_paper.sh
```

The script will:
- ✅ Check for XeLaTeX installation
- ✅ Verify required files exist
- ✅ Compile with proper error handling
- ✅ Process bibliography automatically
- ✅ Generate final PDF

### Method 2: Makefile

```bash
# Full compilation with bibliography
make

# Quick compilation (no bibliography update)
make quick

# Clean auxiliary files
make clean

# View PDF
make view

# Watch for changes
make watch
```

### Method 3: Manual Compilation

```bash
# First compilation
xelatex psi_qrh_lampreia_paper.tex

# Bibliography processing
biber psi_qrh_lampreia_paper
# or: bibtex psi_qrh_lampreia_paper

# Final compilations (resolve references)
xelatex psi_qrh_lampreia_paper.tex
xelatex psi_qrh_lampreia_paper.tex
```

## Paper Structure

### Main Sections

1. **Title Page**: Logo, title, author information
2. **Abstract**: Concise overview of the framework
3. **Introduction**: Lampreia concept and motivation
4. **Mathematical Framework**: ΨQRH equations and formulations
5. **Implementation**: Code architecture and components
6. **Experimental Results**: GLUE benchmarks and efficiency metrics
7. **Key Features**: Technical innovations
8. **Validation**: Testing methodology
9. **Limitations**: Current constraints and future work
10. **Conclusion**: Summary and implications

### Appendices

- **Installation Guide**: Setup instructions
- **Architecture Details**: Component specifications
- **Performance Benchmarks**: Detailed results

## Key Features

### XeLaTeX Advantages

- ✅ **Unicode Support**: Proper rendering of mathematical symbols (Ψ, ℍ, ⊗)
- ✅ **Modern Fonts**: High-quality typography with TeX Gyre fonts
- ✅ **Code Highlighting**: Syntax-colored Python code blocks
- ✅ **Vector Graphics**: TikZ diagrams for architecture visualization
- ✅ **Bibliography**: Automated reference management

### Content Highlights

- 🧮 **Genuine Mathematics**: Proper mathematical notation for ΨQRH framework
- 🏗️ **Architecture Diagrams**: Visual representation of distillation pipeline
- 📊 **Performance Tables**: Comprehensive benchmarking results
- 💻 **Code Examples**: Minted syntax highlighting for Python
- 📚 **Academic References**: Properly formatted bibliography

## Customization

### Font Configuration

The document uses TeX Gyre fonts by default. To change fonts, modify the XeLaTeX preamble:

```latex
% In psi_qrh_lampreia_paper.tex
\setmainfont[Ligatures=TeX]{Your-Font-Family}
\setsansfont[Ligatures=TeX]{Your-Sans-Family}
\setmonofont[Scale=0.8]{Your-Mono-Family}
```

### Color Scheme

Colors are defined in the preamble. Modify for different themes:

```latex
% Custom colors
\definecolor{psiqrhblue}{RGB}{0,102,204}
\definecolor{psiqrhred}{RGB}{204,0,51}
\definecolor{psiqrhgreen}{RGB}{0,153,76}
```

### Bibliography Style

Currently uses `plain` style. Change in the document:

```latex
\bibliographystyle{ieeetr}  % or apa, acm, etc.
```

## Troubleshooting

### Common Issues

**Font Not Found Error:**
```
! The font "TeX Gyre Termes" cannot be found
```
**Solution:** Install TeX Gyre fonts or change to system fonts

**Bibliography Compilation Error:**
```
! LaTeX Error: File `psi_qrh_lampreia_paper.bbl' not found
```
**Solution:** Run `biber psi_qrh_lampreia_paper` or `bibtex psi_qrh_lampreia_paper`

**Missing Packages Error:**
```
! LaTeX Error: File `minted.sty' not found
```
**Solution:** Install additional LaTeX packages or comment out optional packages

### XeLaTeX vs PDFLaTeX

This document requires XeLaTeX due to:
- Unicode characters (Ψ, ℍ, ⊗)
- System font integration
- Advanced typography features

If you must use PDFLaTeX, modify the document to remove XeLaTeX-specific features.

## Output

Successful compilation generates:

- `psi_qrh_lampreia_paper.pdf`: Final document (A4, ~20-30 pages)
- Various auxiliary files (`.aux`, `.log`, `.bbl`, etc.)

## Academic Standards

This paper follows academic publishing standards:

- ✅ **Proper Citations**: IEEE/APA compatible references
- ✅ **Mathematical Notation**: Standard mathematical typesetting
- ✅ **Figure Captions**: Descriptive and informative
- ✅ **Table Formatting**: Professional booktabs styling
- ✅ **Code Listings**: Syntax highlighting and line numbers

## Related Files

- `psi_qrh_benchmark_lampreia.py`: Complete implementation
- `README.md`: GitHub-friendly documentation
- `doe.md`: Original ΨQRH framework specification

## License

This technical paper is part of the ΨQRH Lampreia project, licensed under GNU GPLv3.

---

**Ready to compile?** Run `./compile_paper.sh` and enjoy your professionally typeset technical paper! 📄✨