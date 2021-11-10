using System;
using Microsoft.ML.Data;

namespace App.ML.Objects
{
    public class EmalPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category;
    }
}

