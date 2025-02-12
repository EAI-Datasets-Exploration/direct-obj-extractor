.PHONY: install
install:
	python3 -m pip install .

.PHONY: dev_install
dev_install:
	python3 -m pip install '.[dev,test]'

.PHONY: lint
lint:
	python3 -m pylint direct_obj_extractor/

.PHONY: format
format:
	python3 -m black direct_obj_extractor/

.PHONY: test
test:
	python3 -m pytest test/