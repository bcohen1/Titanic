VENV = titanic

activate:
	$(VENV)\Scripts\activate

init:
	pip install -r requirements.txt

format: activate
	@black .

test: activate
	pytest

build-image: init
	docker build --tag $(VENV) --no-cache .

run-container: build-image
	docker run -d -it -v $(CURDIR)/Docker:/Docker/mount --name $(VENV) --rm $(VENV)

run-script: run-container
	docker exec -it $(VENV) bash -c "python titanic.py; mv submission.csv ./mount"
	docker stop $(VENV)