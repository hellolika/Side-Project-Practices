using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class LeaveAmountResponseV2
    {
        [JsonProperty("LeaveId")]
        public int LeaveId { get; set; }

        [JsonProperty("LeaveType")]
        public string LeaveType { get; set; }

        [JsonProperty("Allowance")]
        public decimal Allowance { get; set; }

        [JsonProperty("Earned")]
        public decimal Earned { get; set; }

        [JsonProperty("Taken")]
        public decimal Taken { get; set; }

        [JsonProperty("Availability")]
        public decimal Availability { get; set; }

        [JsonProperty("Year")]
        public int Year { get; set; }
    }
}
