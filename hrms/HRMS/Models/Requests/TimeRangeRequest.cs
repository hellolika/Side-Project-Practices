using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class TimeRangeRequest
    {
        [JsonProperty("StartDate")]
        public DateTimeOffset StartDate { get; set; }

        [JsonProperty("EndDate")]
        public DateTimeOffset EndDate { get; set; }
    }
}