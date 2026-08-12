using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;
using System.Text.Json.Serialization;

namespace HRMS.Models.Requests
{
    public class EditMemberLeaveRequest
    {
        [JsonProperty("MemberId")] public int MemberId { get; set; }
        [JsonProperty("LeaveTypeId")] public int LeaveTypeId { get; set; }
        [JsonProperty("ReadjustAmount")] public float ReadjustAmount { get; set; }

        public void CheckModels()
        {
            if (MemberId < 0)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Member Id cannot be negative");
            }
            if (LeaveTypeId < 0)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Leave Type Id cannot be negative");
            }
        }
    }
}