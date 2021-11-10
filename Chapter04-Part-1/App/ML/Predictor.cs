using System;
using System.Text.Json;
using App.ML.Base;
using App.ML.Objects;
using Microsoft.ML;

namespace App.ML
{
    public class Predictor : BaseML
    {
        public void Predict(string inputDataFile)
        {
            if (!File.Exists(ModelPath))
            {
                Console.WriteLine($"Failed to find model at {ModelPath}");

                return;
            }

            if (!File.Exists(inputDataFile))
            {
                Console.WriteLine($"Failed to find input data at {inputDataFile}");

                return;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }

            if (mlModel == null)
            {
                Console.WriteLine("Failed to load model");

                return;
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<CarInventory, CarInventoryPrediction>(mlModel);

            var json = File.ReadAllText(inputDataFile);

            var prediction = predictionEngine.Predict(JsonSerializer.Deserialize<CarInventory>(json));

            Console.WriteLine(
                                $"Based on input json:{System.Environment.NewLine}" +
                                $"{json}{System.Environment.NewLine}" +
                                $"The car price is a {(prediction.PredictedLabel ? "good" : "bad")} deal, with a {prediction.Probability:P0} confidence");
        }
    }
}

