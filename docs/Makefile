SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

html:
	sphinx-apidoc -o api/ ../momentGW --force
	sphinx-build -M html . _build

.PHONY: html
