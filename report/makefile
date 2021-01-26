default:
	mkdir output -p
	rm -f output/*.aux
	xelatex -halt-on-error -output-directory output main.tex
	BSTINPUT="out:" TEXMFOUTPUT="out:" bibtex out/main
	xelatex -halt-on-error -output-directory output main.tex
	xelatex -halt-on-error -output-directory output main.tex

watch:
	make
	when-changed main.tex make     # may need to run "sudo pip install when-changed"
	
