SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

html:
	sphinx-build -M html . _build

postprocess:
	# Post-processing generated files:
	# If there is a line with a "Duplicate implicit target name" error, remove it:
	for file in `find _build/html -name "*.html"`; do \
		sed -i '/Duplicate implicit target name/d' $$file; \
	done

docs: html postprocess

clean:
	rm -rf _build autoapi

.PHONY: html postprocess clean
