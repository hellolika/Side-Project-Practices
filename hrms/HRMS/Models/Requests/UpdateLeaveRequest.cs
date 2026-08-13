using System;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class UpdateLeaveRequest
	{
        [JsonProperty("RequestId")]
        public int RequestId { get; set; }

        [JsonProperty("NumberOfDay")]
        public decimal NumberOfDay { get; set; }

        [JsonProperty("StartDate")]
        public DateTimeOffset StartDate { get; set; }

        [JsonProperty("EndDate")]
        public DateTimeOffset EndDate { get; set; }

        [JsonProperty("LeaveType")]
        public int LeaveType { get; set; }

        [JsonProperty("Reason")]
        public string Reason { get; set; }
    }
}

