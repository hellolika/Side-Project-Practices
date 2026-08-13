using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class GetLeaveAmountByMemberIdRequest
    {
        [JsonProperty("MemberId")] public int MemberId { get; set; }

        public void CheckModels()
        {

            if (MemberId < 0)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Member Id cannot be naegative.");
            }

        }
    }
}