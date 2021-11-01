using Microsoft.ML.Data;

namespace Chapter03.App.Ml.Objects
{
    public class EmploymentHistoryPrediction
    {
        [ColumnName("Score")]
        public float DurationInMonths { get; set; }
    }
}