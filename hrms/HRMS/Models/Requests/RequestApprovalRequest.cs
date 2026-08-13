using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Extensions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class RequestApprovalRequest
    {
        [JsonProperty("RequestId")]
        public int RequestId { get; set; }
        
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        [JsonProperty("IsApproved")]
        public int IsApproved { get; set; }

        [JsonProperty("ResponseReason")] public string ResponseReason { get; set; } = "";

        [JsonIgnore]
        public int ApproverId { get; set; }

        public void CheckModels()
        {
            // if(!IsApproved)
            // {
            //     throw new ApiException(ApiErrorEnum.InvalidModelState, "A reason is required for rejecting the request");
            // }
            // if (ResponseReason?.Length > 200)
            // {
            //     throw new ApiException(ApiErrorEnum.InvalidModelState, "Reason must not exceed 200 characters in total");
            // }
        }
    }
}