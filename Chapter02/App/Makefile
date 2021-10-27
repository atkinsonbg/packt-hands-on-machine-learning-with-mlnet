build:
	dotnet build -o BuildOutput

train:
	./BuildOutput/App train Data/sampledata.tsv

predict-p:
	./BuildOutput/App predict "this is a amazing thing, you should get it!"

predict-n:
	./BuildOutput/App predict "quality was very poor!"