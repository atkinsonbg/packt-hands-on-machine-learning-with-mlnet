using Microsoft.ML.Data;

namespace Chapter02.App.Ml.Objects
{
    public class RestaurantPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Label { get; set; }
        public float[] Score { get; set; }
    }
}