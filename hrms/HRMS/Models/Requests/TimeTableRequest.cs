using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class TimeTableRequest
    {
        [JsonProperty("StartDate")]
        public DateTimeOffset StartDate { get; set; }

        [JsonProperty("EndDate")]
        public DateTimeOffset EndDate { get; set; }
    }
}