Compilation instructions

Upload all files, preserving the figures/ directory.
Main document: main.tex
Compiler: XeLaTeX (required by fontspec and Vietnamese text).
Bibliography: BibTeX with the IEEEtran style.

On Overleaf: New Project > Upload Project, then select this ZIP.
Set the compiler to XeLaTeX in project settings and recompile.

For a local TeX Live installation:
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

The manuscript uses Times New Roman when available and Liberation Serif otherwise.
A local PDF compilation has not been performed in the editing environment.
