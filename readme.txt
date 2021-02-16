Create virtual environment in python

bash:

>> python3 -m venv env

Activate the virtual environment

>> . env/bin/activate 

Install all requirements with pip

>> pip install -r requirements.txt

To test the model, run 

>> python main.py --preptrain n --preptest y

To train the model, run

>> python main.py --preptrain y --preptest n