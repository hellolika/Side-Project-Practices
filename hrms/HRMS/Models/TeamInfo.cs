using Newtonsoft.Json;

namespace HRMS.Models
{
    public class TeamInfo
    {
        [JsonProperty("TeamId")]
        public int TeamId { get; set; }
        
        [JsonProperty("DepartmentId")]
        public int DepartmentId { get; set; }
        
        [JsonProperty("DepartmentName")]
        public string DepartmentName { get; set; }

        [JsonProperty("TeamName")]
        public string TeamName { get; set; }

        [JsonProperty("StartTime")]
        public TimeSpan StartTime { get; set; }

        [JsonProperty("EndTime")]
        public TimeSpan EndTime { get; set; }

        [JsonProperty("TotalHour")]
        public decimal TotalHour { get; set; }

        [JsonProperty("IsEnable")]
        public bool IsEnable { get; set; }
    }
}