default:
	mkdir output -p
	rm -f output/*.aux
	xelatex -halt-on-error -output-directory output main-report.tex
	BSTINPUT="output:" TEXMFOUTPUT="output:" bibtex output/main-report
	xelatex -halt-on-error -output-directory output main-report.tex
	xelatex -halt-on-error -output-directory output main-report.tex
	ln -f output/main-report.pdf ./report.pdf
	
	xelatex -halt-on-error -output-directory output main-readme.tex
	ln -f output/main-readme.pdf ./readme.pdf

watch:
	make
	when-changed `ls -1 | grep -P '\.(tex)' | xargs` -c make     # may need to run "sudo pip install when-changed"
	
