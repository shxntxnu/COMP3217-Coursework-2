setup:
	pip3 install -r requirements.txt

run:
	python3 classify.py TrainingData.txt TestingData.txt

print:
	python3 classify.py TrainingData.txt TestingData.txt TestingResults.txt

compare:
	python3 comparison.py

schedule:
	python3 schedule_plot.py
