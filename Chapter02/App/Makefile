build:
	dotnet build -o BuildOutput

train-sample:
	./BuildOutput/App train Data/book_example_data.csv

train-all:
	./BuildOutput/App train Data/all_data.csv

predict-p:
	./BuildOutput/App predict "I will always go to this place to eat, the food is great"

predict-n:
	./BuildOutput/App predict "I will never go to this place ever again, the food is terrible"