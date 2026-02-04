# Makefile for LaTeX document compilation

# Document name (without .tex extension)
DOC = action_memory_implementation
DIAGRAM = network_diagram

# LaTeX compiler
LATEX = pdflatex

# BibTeX compiler (if needed in future)
BIBTEX = bibtex

# Output directory
OUTDIR = .

# Main target - compile both document and diagram
all: $(DOC).pdf $(DIAGRAM).pdf

# Compile main PDF
$(DOC).pdf: $(DOC).tex
	$(LATEX) -output-directory=$(OUTDIR) $(DOC).tex
	$(LATEX) -output-directory=$(OUTDIR) $(DOC).tex  # Run twice for references

# Compile diagram PDF
$(DIAGRAM).pdf: $(DIAGRAM).tex
	$(LATEX) -output-directory=$(OUTDIR) $(DIAGRAM).tex

# Clean auxiliary files
clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
	      $(DOC).bbl $(DOC).blg $(DOC).fls $(DOC).fdb_latexmk \
	      $(DOC).synctex.gz $(DOC).nav $(DOC).snm $(DOC).vrb \
	      $(DOC).bcf $(DOC).run.xml \
	      $(DIAGRAM).aux $(DIAGRAM).log $(DIAGRAM).out

# Clean everything including PDFs
cleanall: clean
	rm -f $(DOC).pdf $(DIAGRAM).pdf

# View PDF (macOS)
view: $(DOC).pdf
	open $(DOC).pdf

# Recompile (clean + build)
rebuild: clean all

.PHONY: all clean cleanall view rebuild
