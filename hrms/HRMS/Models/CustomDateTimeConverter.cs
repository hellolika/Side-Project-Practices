using Newtonsoft.Json.Converters;

namespace HRMS.Models
{
    public class CustomDateTimeConverter : IsoDateTimeConverter
    {
        public CustomDateTimeConverter()
        {
            DateTimeFormat = $"yyyy-MM-dd";
        }
    }
}