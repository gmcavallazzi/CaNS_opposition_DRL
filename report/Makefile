# Makefile for LaTeX document compilation

# Document name (without .tex extension)
DOC = action_memory_implementation

# LaTeX compiler
LATEX = pdflatex

# BibTeX compiler (if needed in future)
BIBTEX = bibtex

# Output directory
OUTDIR = .

# Main target
all: $(DOC).pdf

# Compile PDF
$(DOC).pdf: $(DOC).tex
	$(LATEX) -output-directory=$(OUTDIR) $(DOC).tex
	$(LATEX) -output-directory=$(OUTDIR) $(DOC).tex  # Run twice for references

# Clean auxiliary files
clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
	      $(DOC).bbl $(DOC).blg $(DOC).fls $(DOC).fdb_latexmk \
	      $(DOC).synctex.gz $(DOC).nav $(DOC).snm $(DOC).vrb \
	      $(DOC).bcf $(DOC).run.xml

# Clean everything including PDF
cleanall: clean
	rm -f $(DOC).pdf

# View PDF (macOS)
view: $(DOC).pdf
	open $(DOC).pdf

# Recompile (clean + build)
rebuild: clean all

.PHONY: all clean cleanall view rebuild
