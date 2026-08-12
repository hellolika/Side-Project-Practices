using HRMS.Enum;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class ReClocksResponse
    {
        [JsonProperty("RequestId")]
        public int RequestId { get; set; }

        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        [JsonProperty("Username")]
        public string Username { get; set; }

        [JsonProperty("TeamId")]
        public int TeamId { get; set; }

        [JsonProperty("TeamName")]
        public string TeamName { get; set; }

        [JsonProperty("Date")]
        [JsonConverter(typeof(CustomDateTimeConverter))]
        public DateTime Date { get; set; }

        [JsonProperty("Time")]
        //[JsonConverter(typeof(CustomTimeSpanConverter))]
        public TimeSpan Time { get; set; }

        [JsonProperty("IsClockIn")]
        public bool IsClockIn { get; set; }

        [JsonProperty("Status")]
        public StatusEnum Status { get; set; }

        [JsonProperty("Reason")]
        public string Reason { get; set; }

        [JsonProperty("UpdateBy")]
        public string UpdateBy { get; set; }
    }
}
