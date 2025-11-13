# XeLaTeX Makefile for ΨQRH Lampreia Paper
# Requires: xelatex, biber/bibtex

PAPER = psi_qrh_lampreia_paper
TEX_FILE = $(PAPER).tex
PDF_FILE = $(PAPER).pdf
BIB_FILE = references.bib

# Default target
all: $(PDF_FILE)

# Main compilation with XeLaTeX
$(PDF_FILE): $(TEX_FILE) $(BIB_FILE)
	@echo "Compiling XeLaTeX document..."
	xelatex -interaction=nonstopmode $(TEX_FILE)
	@echo "Processing bibliography..."
	biber $(PAPER) || bibtex $(PAPER)
	@echo "Final compilation..."
	xelatex -interaction=nonstopmode $(TEX_FILE)
	xelatex -interaction=nonstopmode $(TEX_FILE)
	@echo "PDF generated: $(PDF_FILE)"

# Quick compilation (no bibliography)
quick: $(TEX_FILE)
	@echo "Quick XeLaTeX compilation..."
	xelatex -interaction=nonstopmode $(TEX_FILE)
	@echo "Quick PDF generated: $(PDF_FILE)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.bbl *.blg *.bcf *.run.xml
	rm -f *.toc *.lof *.lot *.out *.fdb_latexmk
	rm -f *.fls *.synctex.gz

# Clean all generated files
cleanall: clean
	@echo "Cleaning all generated files..."
	rm -f $(PDF_FILE)

# View PDF (Linux/Mac)
view: $(PDF_FILE)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF_FILE) & \
	elif command -v open > /dev/null; then \
		open $(PDF_FILE) & \
	else \
		echo "Could not find PDF viewer"; \
	fi

# Watch for changes and recompile
watch:
	@echo "Watching for changes... (Ctrl+C to stop)"
	@while true; do \
		inotifywait -qre modify $(TEX_FILE) $(BIB_FILE); \
		make quick; \
	done

# Help target
help:
	@echo "XeLaTeX Makefile for ΨQRH Lampreia Paper"
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Full compilation with bibliography (default)"
	@echo "  quick    - Quick compilation without bibliography updates"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove all generated files"
	@echo "  view     - Open PDF in default viewer"
	@echo "  watch    - Watch for file changes and recompile"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - XeLaTeX (xelatex command)"
	@echo "  - Biber or BibTeX for bibliography"
	@echo "  - Font packages: tex-gyre, fontspec"
	@echo ""
	@echo "Usage:"
	@echo "  make          # Full compilation"
	@echo "  make quick    # Fast preview"
	@echo "  make view     # Open PDF"
	@echo "  make clean    # Clean up"

.PHONY: all quick clean cleanall view watch help