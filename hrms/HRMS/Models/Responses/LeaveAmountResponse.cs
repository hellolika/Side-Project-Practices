using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class LeaveAmountResponse
    {
        [JsonProperty("LeaveId")]
        public int LeaveId { get; set; }

        [JsonProperty("LeaveType")]
        public string LeaveType { get; set; }

        [JsonProperty("RemainingLeaves")]
        public decimal RemainingLeaves { get; set; }

        [JsonProperty("LeavesGranted")]
        public decimal LeavesGranted { get; set; }

        [JsonProperty("Year")]
        public int Year { get; set; }
    }
}