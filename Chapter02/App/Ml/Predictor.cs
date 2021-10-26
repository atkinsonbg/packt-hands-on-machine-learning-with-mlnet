using Chapter02.App.Ml.Base;
using Chapter02.App.Ml.Objects;
using Microsoft.ML;

namespace Chapter02.App.Ml
{
    public class Predictor : BaseML
    {
        public void Predict(string inputData)
        {
            if (!File.Exists(ModelPath))
            {
                Console.WriteLine($"Failed to find model at path {ModelPath}");
                return;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }

            if (mlModel == null)
            {
                Console.WriteLine("Failed to load ML model");
                return;
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<RestaurantFeedback, RestaurantPrediction>(mlModel);

            var prediction = predictionEngine.Predict(new RestaurantFeedback
            {
                Text = inputData
            });

            string predictionOutput = (prediction.Prediction ? "Negative" : "Positive");
            string predictionConfidence = $"{prediction.Probability:P0}";

            Console.WriteLine(
                $"Based on '{inputData}', the feedback is predicted to be:{Environment.NewLine}" +
                $"{predictionOutput} at a {predictionConfidence} confidence."
            );
        }


    }
}