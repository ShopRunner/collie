clean:
	rm -rf dist/

dist: clean
	python setup.py sdist

egg: dist
	python setup.py bdist_egg
