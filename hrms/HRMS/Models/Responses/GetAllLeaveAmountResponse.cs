using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class GetAllLeaveAmountResponse
    {
        public GetAllLeaveAmountResponse()
        {

        }        
        
        public GetAllLeaveAmountResponse(int memberId, string username, int year, List<MemberLeaveAmount> memberLeaves)
        {
            MemberId = memberId;
            Year = year;
            Username = username;
            Leaves = memberLeaves.Select(l => new LeaveAmount(l)).ToList();
        }

        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        [JsonProperty("Username")]
        public string Username { get; set; }

        [JsonProperty("Year")]
        public int Year { get; set; }

        [JsonProperty("Leaves")]
        public List<LeaveAmount> Leaves { get; set; }
    }

    public class LeaveAmount
    {
        public LeaveAmount(MemberLeaveAmount leave)
        {
            LeaveId = leave.LeaveId;
            LeaveType = leave.LeaveType;
            RemainingLeaves = leave.RemainingLeaves;
            LeavesGranted = leave.LeavesGranted;
        }

        [JsonProperty("LeaveId")]
        public int LeaveId { get; set; }

        [JsonProperty("LeaveType")]
        public string LeaveType { get; set; }

        [JsonProperty("RemainingLeaves")]
        public decimal RemainingLeaves { get; set; }

        [JsonProperty("LeavesGranted")]
        public decimal LeavesGranted { get; set; }

    }
}
