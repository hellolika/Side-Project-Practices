using HRMS.Enum;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class AllRequestFormsResponse
    {
        [JsonProperty("TakeLeave")]
        public List<TakeLeave> TakeLeave { get; set; }

        [JsonProperty("ReClock")]
        public List<ReClock> ReClock { get; set; }

        public AllRequestFormsResponse(List<TakeLeave> TakeLeaveResponse, List<ReClock> reClockResponse)
        {
            TakeLeave = TakeLeaveResponse;
            ReClock = reClockResponse;
        }
    }

    public class TakeLeave
    {
        [JsonProperty("RequestId")]
        public int RequestId { get; set; }

        [JsonProperty("LeaveId")]
        public int LeaveId { get; set; }

        [JsonProperty("LeaveType")]
        public string LeaveType { get; set; }
        
        [JsonProperty("NumberOfDay")]
        public decimal NumberOfDay { get; set; }
        
        [JsonProperty("Image")]
        public string Image { get; set; }

        [JsonProperty("StartDate")]
        public DateTime StartDate { get; set; }

        [JsonProperty("EndDate")]
        public DateTime EndDate { get; set; }

        [JsonProperty("IsCancel")]
        public bool IsCancel { get; set; }

        [JsonProperty("Status")]
        public StatusEnum Status { get; set; }

        [JsonProperty("Reason")]
        public string Reason { get; set; }

        [JsonProperty("ResponseReason")]
        public string ResponseReason { get; set; }

        [JsonProperty("SubmittedOn")]
        public DateTime SubmittedOn { get; set; }

        [JsonProperty("Approver")]
        public string Approver { get; set; }
    }

    public class ReClock
    {
        [JsonProperty("RequestId")]
        public int RequestId { get; set; }

        [JsonProperty("Date")]
        [JsonConverter(typeof(CustomDateTimeConverter))]
        public DateTime Date { get; set; }

        [JsonProperty("Time")]
        [JsonConverter(typeof(CustomTimeSpanConverter))]
        public TimeSpan Time { get; set; }

        [JsonProperty("IsClockIn")]
        public bool IsClockIn { get; set; }

        [JsonProperty("Status")]
        public StatusEnum Status { get; set; }
    }
}