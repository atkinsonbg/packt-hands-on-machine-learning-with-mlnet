build:
	dotnet build -o BuildOutput

extract:
	./BuildOutput/App extract fake

train:
	./BuildOutput/App train Data/sampledata.csv

predict-good:
	./BuildOutput/App predict Data/samples/test/good.ps1

predict-bad:
	./BuildOutput/App predict Data/samples/test/bad.ps1
