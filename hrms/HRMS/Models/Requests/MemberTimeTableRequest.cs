using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class MemberTimeTableRequest : TimeTableRequest
    {
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        public MemberTimeTableRequest()
        {
        }

        public MemberTimeTableRequest(TimeTableRequest timeTable, int memberId)
        {
            StartDate = timeTable.StartDate;
            EndDate = timeTable.EndDate;
            MemberId = memberId;
        }
    }
}