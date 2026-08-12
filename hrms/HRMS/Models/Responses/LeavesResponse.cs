using HRMS.Enum;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class LeavesResponse
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

        [JsonProperty("LeaveId")]
        public int LeaveId { get; set; }
        
        [JsonProperty("Image")]
        public string Image { get; set; }

        [JsonProperty("LeaveType")]
        public string LeaveType { get; set; }

        [JsonProperty("StartDate")]
        public DateTime StartDate { get; set; }

        [JsonProperty("EndDate")]
        public DateTime EndDate { get; set; }

        [JsonProperty("Status")]
        public StatusEnum Status { get; set; }

        [JsonProperty("Reason")]
        public string Reason { get; set; }

        [JsonProperty("ResponseReason")]
        public string ResponseReason { get; set; }

        [JsonProperty("SubmittedOn")]
        public DateTime SubmittedOn { get; set; }

        [JsonProperty("NumberOfDay")]
        public decimal NumberOfDay { get; set; }

        [JsonProperty("UpdateBy")]
        public string UpdateBy { get; set; }

        [JsonProperty("IsCancel")]
        public bool IsCancel { get; set; }
    }
}
