build:
	dotnet build -o BuildOutput

train-sample:
	./BuildOutput/App train Data/sampledata.tsv

train-all:
	./BuildOutput/App train Data/all_data.csv

predict-p:
	./BuildOutput/App predict "this is a great product thing, you should get it!"

predict-n:
	./BuildOutput/App predict "I will never go to this place ever again, the food is terrible"