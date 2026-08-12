using Newtonsoft.Json;

namespace HRMS.Models
{
    public class LeaveType
    {
        [JsonProperty("TypeId")]
        public int TypeId { get; set; }

        [JsonProperty("Type")]
        public string Type { get; set; }

        [JsonProperty("DefaultLeavesGranted")]
        public decimal DefaultLeavesGranted { get; set; }

        [JsonProperty("IsEnable")]
        public bool IsEnable { get; set; }

        [JsonProperty("IsLimited")]
        public bool IsLimited { get; set; }
    }
}