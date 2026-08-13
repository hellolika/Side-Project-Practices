using Newtonsoft.Json;

namespace HRMS.Models
{
    public class CustomTimeSpanConverter : JsonConverter<TimeSpan>
    {
        private readonly TimeSpan _localOffSet = TimeZoneInfo.Local.GetUtcOffset(DateTime.Now);

        public override void WriteJson(JsonWriter writer, TimeSpan value, JsonSerializer serializer)
        {
            var timespanFormatted = GetFormattedTimeSpan(value.Add(-_localOffSet));
            writer.WriteValue(timespanFormatted);
        }

        public override TimeSpan ReadJson(JsonReader reader, Type objectType, TimeSpan existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string text = ((string)reader.Value).Replace("+", "");
            _ = TimeSpan.TryParse(text, out TimeSpan parsedTimeSpan);
            return parsedTimeSpan;
        }

        private static string GetFormattedTimeSpan(TimeSpan time)
        {
            return (time < TimeSpan.Zero ? "-" : "") + time.ToString(@"hh\:mm");
        }
    }
}
