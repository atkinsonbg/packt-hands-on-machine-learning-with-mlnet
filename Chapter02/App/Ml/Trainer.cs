using Chapter02.App.Ml.Base;
using Chapter02.App.Ml.Objects;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace Chapter02.App.Ml
{
    public class Trainer : BaseML
    {
        public void Train(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Could not find the training file {trainingFileName}");
                return;
            }

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<RestaurantFeedback>(trainingFileName);

            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(RestaurantFeedback.Label))
                    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Text", outputColumnName: "Text"))
                    .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: new string[] {
                        "Text"
                    }));

            var trainingPipeline = dataProcessPipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10)))
                                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);

            Console.WriteLine("=============== Starting 10 fold cross validation ===============");
            var crossValResults = mlContext.MulticlassClassification.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numberOfFolds: 10, labelColumnName: "Label");
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);
            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average(); Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###} ");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###} ");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###} ");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###} ");
            Console.WriteLine($"*************************************************************************************************************");
        }
    }
}