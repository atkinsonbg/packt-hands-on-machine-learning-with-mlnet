using System;
using System.IO;
using System.Text;

using Chapter03.App.Common;
using Chapter03.App.ML.Base;

namespace Chapter03.App.ML
{
    public class FeatureExtractor : BaseML
    {
        public void Extract(string folderPath)
        {
            string folderPathGood = "../App/Data/samples/clean";
            string folderPathBad = "../App/Data/samples/malicious";

            using (var streamWriter =
                new StreamWriter(Path.Combine(AppContext.BaseDirectory, $"../../App/Data/{Constants.SAMPLE_DATA}")))
            {
                var goodFiles = Directory.GetFiles(folderPathGood);

                foreach (var file in goodFiles)
                {
                    var strings = GetStrings(File.ReadAllBytes(file));

                    streamWriter.WriteLine($"True\t{strings}");
                }

                var badFiles = Directory.GetFiles(folderPathBad);

                foreach (var file in goodFiles)
                {
                    var strings = GetStrings(File.ReadAllBytes(file));

                    streamWriter.WriteLine($"False\t{strings}");
                }
            }
        }
    }
}