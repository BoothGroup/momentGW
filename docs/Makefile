SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

html:
	sphinx-apidoc -o api/ ../momentGW --force --separate
	sphinx-build -M html . _build

clean:
	rm -rf _build api

.PHONY: html
