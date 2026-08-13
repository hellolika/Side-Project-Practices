using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Extensions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class CancelLeaveRequest
    {
        [JsonProperty("RequestId")]
        public int RequestId { get; set; }

        internal int MemberId { get; set; }
    }
}