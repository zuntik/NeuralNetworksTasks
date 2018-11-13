TEX=pdflatex
BIBER=biber

make:
	cd ./cover && $(TEX) cover.tex
	$(TEX) report.tex

pdf: make

clean:
	rm -rf report.pdf report.aux report.toc report.out report.log report.dvi cover/cover.pdf cover/cover.aux cover/cover.log *.bbl *.bcf *.blg *.xml
