using HRMS.Enum;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class ClockStatusResponse
    {
        [JsonProperty("ClockInStatus")]
        public StatusEnum ClockInStatus { get; set; }

        [JsonProperty("ClockOutStatus")]
        public StatusEnum ClockOutStatus { get; set; }
        
        [JsonProperty("ClockInTime")]
        public TimeSpan? ClockInTime { get; set; }

        [JsonProperty("ClockOutTime")]
        public TimeSpan? ClockOutTime { get; set; }

        [JsonProperty("IsOnLeave")]
        public bool IsOnLeave { get; set; }
    }
}