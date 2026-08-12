using HRMS.Models.Responses;
using Newtonsoft.Json;

namespace HRMS.Models
{
    public class MemberLeaveAmount : LeaveAmountResponse
    {
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        [JsonProperty("Username")]
        public string Username { get; set; }

        [JsonProperty("ModifyBy")]
        public int ModifyBy { get; set; }

        [JsonProperty("ModifyOn")]
        public DateTime ModifyOn { get; set; }

    }
}
