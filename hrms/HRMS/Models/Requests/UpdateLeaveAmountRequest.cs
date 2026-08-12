using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class UpdateLeaveAmountRequest
    {
        [JsonIgnore]
        public int ModifierId { get; set; }

        [JsonProperty("MemberId")]
        public int MemberId { get; set; }
        
        [JsonProperty("LeaveType")]
        public int LeaveType { get; set; }
        
        [JsonProperty("Year")]
        public int Year { get; set; }
        
        [JsonProperty("LeaveAmount")]
        public decimal LeaveAmount { get; set; }
    }
}
