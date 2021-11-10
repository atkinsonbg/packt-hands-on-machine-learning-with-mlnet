using System;
using App.ML.Base;
using App.ML.Objects;
using Microsoft.ML;

namespace App.ML
{
    public class Trainer : BaseML
    {
        public void Train(string trainingFileName, string testFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName}");

                return;
            }

            if (!File.Exists(testFileName))
            {
                Console.WriteLine($"Failed to find test data file ({testFileName}");

                return;
            }

            var trainingDataView = mlContext.Data.LoadFromTextFile<Email>(trainingFileName, ',', hasHeader: false);

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(Email.Category), outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Email.Subject), outputColumnName: "SubjectFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Email.Body), outputColumnName: "BodyFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Email.Sender), outputColumnName: "SenderFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "SubjectFeaturized", "BodyFeaturized", "SenderFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            var trainingPipeline = dataProcessPipeline
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedModel = trainingPipeline.Fit(trainingDataView);
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            var testDataView = mlContext.Data.LoadFromTextFile<Email>(testFileName, ',', hasHeader: false);

            var modelMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));

            Console.WriteLine($"MicroAccuracy: {modelMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"MacroAccuracy: {modelMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"LogLoss: {modelMetrics.LogLoss:#.###}");
            Console.WriteLine($"LogLossReduction: {modelMetrics.LogLossReduction:#.###}");
        }
    }
}

